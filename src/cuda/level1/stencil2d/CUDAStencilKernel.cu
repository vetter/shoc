
//
// workaround for CUDA 2.3a + gcc 4.2.1 problem 
// (undefined __sync_fetch_and_add) on Snow Leopard
//
#if defined(__APPLE__)
#if _GLIBCXX_ATOMIC_BUILTINS == 1
#undef _GLIBCXX_ATOMIC_BUILTINS
#endif // _GLIBCXX_ATOMIC_BUILTINS
#endif // __APPLE__

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "CUDAStencil.cpp"



__device__
int
ToGlobalRow( int gidRow, int lidRow )
{
    return gidRow * LROWS + lidRow;
}

__device__
int
ToGlobalCol( int gidCol, int lidCol )
{
    return gidCol * LCOLS + lidCol;
}

__device__
int
ToFlatIdx( int row, int col, int rowWidth )
{
    // assumes input coordinates and dimensions are logical (without halo)
    // and a halo width of 1
    return (row+1)*(rowWidth + 2) + (col+1);
}


template<class T>
__global__
void
StencilKernel( T* data,
                T* newData,
                int pad,
                T wCenter,
                T wCardinal,
                T wDiagonal )
{
    // determine our location in the coordinate system
    // see the comment in operator() at the definition of the dimGrid
    // and dimBlock dim3s to understand why .x == row and .y == column.
    int gidRow = blockIdx.x;
    int gidCol = blockIdx.y;
    // int gszRow = gridDim.x;
    int gszCol = gridDim.y;
    int lidRow = threadIdx.x;
    int lidCol = threadIdx.y;

    // determine our logical global data coordinates (without halo)
    int gRow = ToGlobalRow( gidRow, lidRow );
    int gCol = ToGlobalCol( gidCol, lidCol );

    // determine pitch of rows (without halo)
    int nCols = gszCol * LCOLS + 2;     // assume halo is there for computing padding
    int nPaddedCols = nCols + (((nCols % pad) == 0) ? 0 : (pad - (nCols % pad)));
    int gRowWidth = nPaddedCols - 2;    // remove the halo


    // determine our coodinate in the flattened data (with halo)
    int gidx = ToFlatIdx( gRow, gCol, gRowWidth );

    // copy my global data item to a shared local buffer
    // sh is (LROWS + 2) x (LCOLS + 2) values
    // (i.e., it is same size as our local block but with halo of width 1)
    __shared__ T sh[LROWS+2][LCOLS+2];
    sh[lidRow+1][lidCol+1] = data[gidx];

    // copy halo data
    // We follow the approach of Micikevicius (NVIDIA) from the
    // GPGPU-2 Workshop, 3/8/2009.
    // We leave many threads idle while those along two of the edges
    // copy the boundary data for all four edges. This seems to be
    // a performance win even with the idle threads because it 
    // limits the branching logic.
    if( lidRow == 0 )
    {
        sh[0][lidCol+1] = data[ToFlatIdx(gRow-1, gCol, gRowWidth)];
        sh[LROWS+1][lidCol+1] = data[ToFlatIdx(gRow+LROWS, gCol, gRowWidth)];
    }
    if( lidCol == 0 )
    {
        sh[lidRow+1][0] = data[ToFlatIdx(gRow, gCol-1, gRowWidth)];
        sh[lidRow+1][LCOLS+1] = data[ToFlatIdx(gRow, gCol+LCOLS, gRowWidth)];
    }
    if( (lidRow == 0) && (lidCol == 0) )
    {
        // since we are doing 9-pt stencil, we have to copy corner elements.
        // Note: stencil used by Micikevicius did not use 'diagonals' - 
        // in 2D, it would be a 5-pt stencil. But these loads are costly.
        sh[0][0] = data[ToFlatIdx(gRow-1,gCol-1,gRowWidth)];
        sh[LROWS+1][0] = data[ToFlatIdx(gRow+LROWS,gCol-1, gRowWidth)];
        sh[0][LCOLS+1] = data[ToFlatIdx(gRow-1,gCol+LCOLS,gRowWidth)];
        sh[LROWS+1][LCOLS+1] = data[ToFlatIdx(gRow+LROWS,gCol+LCOLS, gRowWidth)];
    }

    // let all those loads finish
    __syncthreads();

    // do my part of the smoothing operation
    T centerValue = sh[lidRow+1][lidCol+1];
    T cardinalValueSum = sh[lidRow][lidCol+1] +
                        sh[lidRow+2][lidCol+1] +
                        sh[lidRow+1][lidCol] +
                        sh[lidRow+1][lidCol+2];
    T diagonalValueSum = sh[lidRow][lidCol] +
                        sh[lidRow][lidCol+2] +
                        sh[lidRow+2][lidCol] +
                        sh[lidRow+2][lidCol+2];

    newData[gidx] = wCenter * centerValue +
            wCardinal * cardinalValueSum + 
            wDiagonal * diagonalValueSum;
}


template <class T>
void
CUDAStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    // assume a 1-wide halo
    size_t gRows = mtx.GetNumRows() - 2;
    size_t gCols = mtx.GetNumColumns() - 2;
    assert( gRows % LROWS == 0 );
    assert( gCols % LCOLS == 0 );

    // Note: this is confusing.  C/C++ code on the host and CUDA C on
    // the device use row-major ordering where the first dimension is
    // the row and the second is the column.  In a dim3, the constituent
    // items are named .x, .y, and .z.  Normally, x is considered 
    // horizontal (which would correspond to column position), y is 
    // vertical (which would correspond to row position).  We use
    //   .x == row (first dimension)
    //   .y == column (second dimension)
    // 
    dim3 dimGrid( gRows / LROWS, gCols / LCOLS );
    dim3 dimBlock( LROWS, LCOLS );

    // size of data to transfer to/from device - assume 1-wide halo
    size_t matDataSize = mtx.GetDataSize();
    size_t localDataSize = sizeof(T) * (dimBlock.x + 2) * (dimBlock.y + 2);
    T* da = NULL;
    T* db = NULL;

    // allocate space on device in global memory
    cudaMalloc( (void**)&da, matDataSize );
    cudaMalloc( (void**)&db, matDataSize );

    // copy initial data to global memory
    T* currData = da;
    T* newData = db;
    cudaMemcpy( currData, mtx.GetFlatData(), matDataSize, cudaMemcpyHostToDevice );

    // copy the halo from the initial buffer into the second buffer
    // Note: when doing local iterations, these values do not change 
    // but they can change in the MPI version after an inter-process
    // halo exchange.
    //
    // copy the sides with contiguous data
    size_t rowExtent = mtx.GetNumPaddedColumns() * sizeof(T);
    cudaMemcpy2D( newData,      // destination
                    rowExtent,  // destination pitch
                    currData,   // source
                    rowExtent,  // source pitch
                    rowExtent,  // width of data to transfer (bytes)
                    1,          // height of data to transfer (rows)
                    cudaMemcpyDeviceToDevice );
    cudaMemcpy2D( newData + (mtx.GetNumRows() - 1) * mtx.GetNumPaddedColumns(),      // destination
                    rowExtent,  // destination pitch
                    currData + (mtx.GetNumRows() - 1) * mtx.GetNumPaddedColumns(),   // source
                    rowExtent,  // source pitch
                    rowExtent,  // width of data to transfer (bytes)
                    1,          // height of data to transfer (rows)
                    cudaMemcpyDeviceToDevice );

    // copy the non-contiguous data
    cudaMemcpy2D( newData,      // destination
                    rowExtent,  // destination pitch
                    currData,   // source
                    rowExtent,  // source pitch
                    sizeof(T),  // width of data to transfer (bytes)
                    mtx.GetNumRows(),      // height of data to transfer (rows)
                    cudaMemcpyDeviceToDevice );
    cudaMemcpy2D( newData + (mtx.GetNumColumns() - 1),      // destination
                    rowExtent,  // destination pitch
                    currData + (mtx.GetNumColumns() - 1),   // source
                    rowExtent,  // source pitch
                    sizeof(T),  // width of data to transfer (bytes)
                    mtx.GetNumRows(),      // height of data to transfer (rows)
                    cudaMemcpyDeviceToDevice );

    // run the CUDA kernel
    for( unsigned int iter = 0; iter < nIters; iter++ )
    {
        this->DoPreIterationWork( currData,
                                    newData,
                                    mtx,
                                    iter );

        // do the stencil operation
        StencilKernel<<<dimGrid, dimBlock, localDataSize>>>( currData, 
            newData, 
            mtx.GetPad(),
            this->wCenter, 
            this->wCardinal, 
            this->wDiagonal );

        // swap our notion of which buffer holds the "real" data
        if( currData == da )
        {
            currData = db;
            newData = da;
        }
        else
        {
            currData = da;
            newData = db;
        }
    }

    // get the final result
    cudaMemcpy( mtx.GetFlatData(), currData, matDataSize, cudaMemcpyDeviceToHost );

    // clean up CUDA
    cudaFree( da );
    cudaFree( db );
}


// Ensure our template classes get instantiated for the types needed by
// the rest of the code.
// Note that we are instantiating objects here.  Some of the other
// SHOC benchmarks also need to force template instantiations for specific
// types, but they are using function templates.  This mechanism (putting
// them in a function) will not work if nvcc is using more recent version of
// g++ underneath.  Instead, just declare the function specialization outside
// of any function with a 'template' keyword like so:
// 
// template void Func( int, int, int, float*, int );
// template void Func( int, int, int, double*, int );
//
void
EnsureStencilInstantiation( void )
{
    CUDAStencil<float> csf( 0, 0, 0, 0 );
    Matrix2D<float> mf( 2, 2 );
    csf( mf, 0 );

    CUDAStencil<double> csd( 0, 0, 0, 0 );
    Matrix2D<double> md( 2, 2 );
    csd( md, 0 );
}


