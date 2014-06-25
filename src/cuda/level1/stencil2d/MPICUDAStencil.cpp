#include "mpi.h"
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include "MPICUDAStencil.h"
#include "cuda_runtime.h"


template<class T>
MPICUDAStencil<T>::MPICUDAStencil( T _wCenter,
                                    T _wCardinal,
                                    T _wDiagonal,
                                    size_t _lRows,
                                    size_t _lCols,
                                    size_t _mpiGridRows,
                                    size_t _mpiGridCols,
                                    unsigned int _nItersPerHaloExchange,
                                    int _deviceIdx,
                                    bool _dumpData )
  : CUDAStencil<T>( _wCenter,
                    _wCardinal,
                    _wDiagonal,
                    _lRows,
                    _lCols,
                    _deviceIdx ),
    MPI2DGridProgram<T>( _mpiGridRows,
                    _mpiGridCols,
                    _nItersPerHaloExchange ),
    dumpData( _dumpData )
{
    if( dumpData )
    {
        std::ostringstream fnamestr;
        fnamestr << "cuda." << std::setw( 4 ) << std::setfill('0') << this->GetCommWorldRank();
        ofs.open( fnamestr.str().c_str() );
    }
}


template<class T>
void
MPICUDAStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    if( this->ParticipatingInProgram() )
    {
        // we need to do a halo exchange before our first push of
        // data onto the device
        if( dumpData )
        {
            this->DumpData( ofs, mtx, "before initial halo exchange" );
        }
        this->DoHaloExchange( mtx );
        if( dumpData )
        {
            this->DumpData( ofs, mtx, "after initial halo exchange" );
        }

        // apply the operator
        CUDAStencil<T>::operator()( mtx, nIters );
        if( dumpData )
        {
            this->DumpData( ofs, mtx, "after all iterations" );
        }
    }
    MPI_Barrier( MPI_COMM_WORLD );
}


template<class T>
void
MPICUDAStencil<T>::DoPreIterationWork( T* currBuf,  // in device global memory
                                        T* altBuf,  // in device global memory
                                        Matrix2D<T>& mtx,
                                        unsigned int iter )
{
    // do the halo exchange at desired frequency
    // note that we *do not* do the halo exchange here before the
    // first iteration, because we did it already (before we first
    // pushed the data onto the device) in our operator() method.
    unsigned int haloWidth = this->GetNumberIterationsPerHaloExchange();
    if( (iter > 0) && (iter % haloWidth) == 0 )
    {
        unsigned int nRows = mtx.GetNumRows();
        unsigned int nCols = mtx.GetNumColumns();
        unsigned int nPaddedCols = mtx.GetNumPaddedColumns();
        T* flatData = mtx.GetFlatData();

        size_t nsDataItemCount = haloWidth * nPaddedCols;
        size_t ewDataItemCount = haloWidth * nRows;
        size_t nsDataSize = nsDataItemCount * sizeof(T);
        size_t ewDataSize = ewDataItemCount * sizeof(T);

        //
        // read current data off device
        // we only read halo, and only for sides where we have a neighbor
        //
        if( this->HaveNorthNeighbor() )
        {
            // north data is contiguous - copy directly into matrix
            cudaMemcpy( flatData + (haloWidth * nPaddedCols),   // dest
                        currBuf + (haloWidth * nPaddedCols),     // src
                        nsDataSize,                 // amount to transfer
                        cudaMemcpyDeviceToHost );   // direction
        }

        if( this->HaveSouthNeighbor() )
        {
            // south data is contiguous - copy directly into matrix
            cudaMemcpy( flatData + ((nRows - 2*haloWidth)*nPaddedCols),   // dest
                        currBuf + ((nRows - 2*haloWidth)*nPaddedCols),    // src
                        nsDataSize,                 // amount to transfer
                        cudaMemcpyDeviceToHost );   // direction
        }

        if( this->HaveEastNeighbor() )
        {
            // east data is non-contiguous - but CUDA has a strided read
            cudaMemcpy2D( flatData + (nCols - 2*haloWidth), // dest
                            nPaddedCols * sizeof(T),        // dest pitch
                            currBuf + (nCols - 2*haloWidth),    // src
                            nPaddedCols * sizeof(T),        // src pitch
                            haloWidth * sizeof(T),          // width of data to transfer (bytes)
                            nRows,                          // height of data to transfer (rows)
                            cudaMemcpyDeviceToHost );       // transfer direction
        }

        if( this->HaveWestNeighbor() )
        {
            // west data is non-contiguous - but CUDA has a strided read
            cudaMemcpy2D( flatData + haloWidth,         // dest
                            nPaddedCols * sizeof(T),    // dest pitch
                            currBuf + haloWidth,        // src
                            nPaddedCols * sizeof(T),    // src pitch
                            haloWidth * sizeof(T),      // width of data to transfer (bytes)
                            nRows,          // height of data to transfer (rows)
                            cudaMemcpyDeviceToHost );   // transfer direction

        }


        //
        // do the actual halo exchange
        //
        if( dumpData )
        {
            this->DumpData( ofs, mtx, "before halo exchange" );
        }
        this->DoHaloExchange( mtx );
        if( dumpData )
        {
            this->DumpData( ofs, mtx, "after halo exchange" );
        }

        //
        // push updated data back onto device
        // we only write halo, and only for sides where we have a neighbor
        //
        if( this->HaveNorthNeighbor() )
        {
            // north data is contiguous - copy directly from matrix
            cudaMemcpy( currBuf,                    // dest
                        flatData,                   // src
                        nsDataSize,                 // amount to transfer
                        cudaMemcpyHostToDevice );   // direction
        }

        if( this->HaveSouthNeighbor() )
        {
            // south data is contiguous - copy directly from matrix
            cudaMemcpy( currBuf + ((nRows - haloWidth)*nPaddedCols),    // dest
                        flatData + ((nRows - haloWidth)*nPaddedCols),   // src
                        nsDataSize,                 // amount to transfer
                        cudaMemcpyHostToDevice );   // direction
        }

        if( this->HaveEastNeighbor() )
        {
            // east data is non-contiguous - but CUDA has a strided write
            cudaMemcpy2D( currBuf + (nCols - haloWidth),  // dest
                            nPaddedCols * sizeof(T),              // dest pitch
                            flatData + (nCols - haloWidth), // src
                            nPaddedCols * sizeof(T),              // src pitch
                            haloWidth * sizeof(T),          // width of data to transfer (bytes)
                            nRows,                          // height of data to transfer (rows)
                            cudaMemcpyHostToDevice );       // transfer direction
        }

        if( this->HaveWestNeighbor() )
        {
            // west data is non-contiguous - but CUDA has a strided write
            cudaMemcpy2D( currBuf,                      // dest
                            nPaddedCols * sizeof(T),          // dest pitch
                            flatData,                   // src
                            nPaddedCols * sizeof(T),          // src pitch
                            haloWidth * sizeof(T),      // width of data to transfer (bytes)
                            nRows,          // height of data to transfer (rows)
                            cudaMemcpyHostToDevice );   // transfer direction

        }


        // we have changed the local halo values on the device
        // we need to update the local 1-wide halo in the alt buffer
        // note we only need to update the 1-wide halo here, even if
        // our real halo width is larger
        size_t rowExtent = mtx.GetNumPaddedColumns() * sizeof(T);
        cudaMemcpy2D( altBuf,      // destination
                        rowExtent,  // destination pitch
                        currBuf,   // source
                        rowExtent,  // source pitch
                        rowExtent,  // width of data to transfer (bytes)
                        1,          // height of data to transfer (rows)
                        cudaMemcpyDeviceToDevice );
        cudaMemcpy2D( altBuf + (mtx.GetNumRows() - 1) * mtx.GetNumPaddedColumns(),      // destination
                        rowExtent,  // destination pitch
                        currBuf + (mtx.GetNumRows() - 1) * mtx.GetNumPaddedColumns(),   // source
                        rowExtent,  // source pitch
                        rowExtent,  // width of data to transfer (bytes)
                        1,          // height of data to transfer (rows)
                        cudaMemcpyDeviceToDevice );

        // copy the non-contiguous data
        cudaMemcpy2D( altBuf,      // destination
                        rowExtent,  // destination pitch
                        currBuf,   // source
                        rowExtent,  // source pitch
                        sizeof(T),  // width of data to transfer (bytes)
                        mtx.GetNumRows(),      // height of data to transfer (rows)
                        cudaMemcpyDeviceToDevice );
        cudaMemcpy2D( altBuf + (mtx.GetNumColumns() - 1),      // destination
                        rowExtent,  // destination pitch
                        currBuf + (mtx.GetNumColumns() - 1),   // source
                        rowExtent,  // source pitch
                        sizeof(T),  // width of data to transfer (bytes)
                        mtx.GetNumRows(),      // height of data to transfer (rows)
                        cudaMemcpyDeviceToDevice );
    }
}

