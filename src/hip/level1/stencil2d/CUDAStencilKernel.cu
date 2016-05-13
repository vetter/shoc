
//
// workaround for CUDA 2.3a + gcc 4.2.1 problem
// (undefined __sync_fetch_and_add) on Snow Leopard
//
// #if defined(__APPLE__)
// #if _GLIBCXX_ATOMIC_BUILTINS == 1
// #undef _GLIBCXX_ATOMIC_BUILTINS
// #endif // _GLIBCXX_ATOMIC_BUILTINS
// #endif // __APPLE__

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cudacommon.h"
#include "CUDAStencil.cpp"
using std::cout;
using std::cin;

//
// We are using the "trick" illustrated by the NVIDIA simpleTemplates example
// for accessing dynamically-allocated shared memory from a templatized
// function.  The strategy uses a templatized struct with specialized
// accessor functions that declare the actual symbol with the type in
// their type name (to avoid naming conflicts).
//
template<typename T>
struct SharedMemory
{
    // Should never be instantiated.
    // We enforce this at compile time.
    __device__ T* GetPointer( void )
    {
        extern __device__ void error( void );
        error();
        return NULL;
    }
};

// specializations for types we use
template<>
struct SharedMemory<float>
{
    __device__ float* GetPointer( void )
    {
        extern __shared__ float sh_float[];
        // printf( "sh_float=%p\n", sh_float );
        return sh_float;
    }
};

template<>
struct SharedMemory<double>
{
    __device__ double* GetPointer( void )
    {
        extern __shared__ double sh_double[];
        // printf( "sh_double=%p\n", sh_double );
        return sh_double;
    }
};




__device__
int
ToGlobalRow( int gidRow, int lszRow, int lidRow )
{
    return gidRow * lszRow + lidRow;
}

__device__
int
ToGlobalCol( int gidCol, int lszCol, int lidCol )
{
    return gidCol * lszCol + lidCol;
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
                int alignment,
                int nStripItems,
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
    int lszRow = nStripItems;
    int lszCol = blockDim.y;

    // determine our logical global data coordinates (without halo)
    int gRow = ToGlobalRow( gidRow, lszRow, lidRow );
    int gCol = ToGlobalCol( gidCol, lszCol, lidCol );

    // determine pitch of rows (without halo)
    int nCols = gszCol * lszCol + 2;     // assume halo is there for computing padding
    int nPaddedCols = nCols + (((nCols % alignment) == 0) ? 0 : (alignment - (nCols % alignment)));
    int gRowWidth = nPaddedCols - 2;    // remove the halo

    // Copy my global data item to a shared local buffer.
    // That local buffer is passed to us.
    // We assume it is large enough to hold all the data computed by
    // our thread block, plus a halo of width 1.
    SharedMemory<T> shobj;
    T* sh = shobj.GetPointer();
    int lRowWidth = lszCol;
    for( int i = 0; i < (lszRow + 2); i++ )
    {
        int lidx = ToFlatIdx( lidRow - 1 + i, lidCol, lRowWidth );
        int gidx = ToFlatIdx( gRow - 1 + i, gCol, gRowWidth );
        sh[lidx] = data[gidx];
    }

    // Copy the "left" and "right" halo rows into our local memory buffer.
    // Only two threads are involved (first column and last column).
    if( lidCol == 0 )
    {
        for( int i = 0; i < (lszRow + 2); i++ )
        {
            int lidx = ToFlatIdx(lidRow - 1 + i, lidCol - 1, lRowWidth );
            int gidx = ToFlatIdx(gRow - 1 + i, gCol - 1, gRowWidth );
            sh[lidx] = data[gidx];
        }
    }
    else if( lidCol == (lszCol - 1) )
    {
        for( int i = 0; i < (lszRow + 2); i++ )
        {
            int lidx = ToFlatIdx(lidRow - 1 + i, lidCol + 1, lRowWidth );
            int gidx = ToFlatIdx(gRow - 1 + i, gCol + 1, gRowWidth );
            sh[lidx] = data[gidx];
        }
    }

    // let all those loads finish
    __syncthreads();

    // do my part of the smoothing operation
    for( int i = 0; i < lszRow; i++ )
    {
        int cidx  = ToFlatIdx( lidRow     + i, lidCol    , lRowWidth );
        int nidx  = ToFlatIdx( lidRow - 1 + i, lidCol    , lRowWidth );
        int sidx  = ToFlatIdx( lidRow + 1 + i, lidCol    , lRowWidth );
        int eidx  = ToFlatIdx( lidRow     + i, lidCol + 1, lRowWidth );
        int widx  = ToFlatIdx( lidRow     + i, lidCol - 1, lRowWidth );
        int neidx = ToFlatIdx( lidRow - 1 + i, lidCol + 1, lRowWidth );
        int seidx = ToFlatIdx( lidRow + 1 + i, lidCol + 1, lRowWidth );
        int nwidx = ToFlatIdx( lidRow - 1 + i, lidCol - 1, lRowWidth );
        int swidx = ToFlatIdx( lidRow + 1 + i, lidCol - 1, lRowWidth );

        T centerValue = sh[cidx];
        T cardinalValueSum = sh[nidx] + sh[sidx] + sh[eidx] + sh[widx];
        T diagonalValueSum = sh[neidx] + sh[seidx] + sh[nwidx] + sh[swidx];

        newData[ToFlatIdx(gRow + i, gCol, gRowWidth)] = wCenter * centerValue +
                wCardinal * cardinalValueSum +
                wDiagonal * diagonalValueSum;
    }
}



template <class T>
void
CUDAStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    // assume a 1-wide halo
    size_t gRows = mtx.GetNumRows() - 2;
    size_t gCols = mtx.GetNumColumns() - 2;
    assert( gRows % lRows == 0 );
    assert( gCols % lCols == 0 );

    // Note: this is confusing.  C/C++ code on the host and CUDA C on
    // the device use row-major ordering where the first dimension is
    // the row and the second is the column.  In a dim3, the constituent
    // items are named .x, .y, and .z.  Normally, x is considered
    // horizontal (which would correspond to column position), y is
    // vertical (which would correspond to row position).  We use
    //   .x == row (first dimension)
    //   .y == column (second dimension)
    //
    // Since each GPU thread is responsible for a strip of data
    // from the original, our index space is scaled smaller in
    // one dimension relative to the actual data
    dim3 dimGrid( gRows / lRows, gCols / lCols );
    dim3 dimBlock( 1, lCols );

    // size of data to transfer to/from device - assume 1-wide halo
    size_t matDataSize = mtx.GetDataSize();
    size_t localDataSize = sizeof(T) * (lRows + 2) * (lCols + 2);
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
            lRows,
            this->wCenter,
            this->wCardinal,
            this->wDiagonal );

        CHECK_CUDA_ERROR();

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
    CUDAStencil<float> csf( 0, 0, 0, 0, 0, 0 );
    Matrix2D<float> mf( 2, 2 );
    csf( mf, 0 );

    CUDAStencil<double> csd( 0, 0, 0, 0, 0, 0 );
    Matrix2D<double> md( 2, 2 );
    csd( md, 0 );
}


