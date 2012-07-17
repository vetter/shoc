#include "mpi.h"
#include <sstream>
#include <iomanip>
#include <cassert>
#include "MPIOpenCLStencil.h"



template<class T>
MPIOpenCLStencil<T>::MPIOpenCLStencil( T _wCenter,
                                    T _wCardinal,
                                    T _wDiagonal,
                                    size_t _lRows,
                                    size_t _lCols,
                                    size_t _mpiGridRows,
                                    size_t _mpiGridCols,
                                    unsigned int _nItersPerHaloExchange,
                                    cl::Device& _dev,
                                    cl::Context& _ctx,
                                    cl::CommandQueue& _queue,
                                    bool _dumpData )
  : OpenCLStencil<T>( _wCenter,
                    _wCardinal,
                    _wDiagonal,
                    _lRows,
                    _lCols,
                    _dev,
                    _ctx,
                    _queue ),
    MPI2DGridProgram<T>( _mpiGridRows,
                    _mpiGridCols,
                    _nItersPerHaloExchange ),
    dumpData( _dumpData ),
    eData( NULL ),
    wData( NULL )
{
    if( dumpData )      
    {
        std::ostringstream fnamestr;
        fnamestr << "ocl." << std::setw( 4 ) << std::setfill('0') << this->GetCommWorldRank();
        ofs.open( fnamestr.str().c_str() );
    }
}

template<class T>
MPIOpenCLStencil<T>::~MPIOpenCLStencil( void )
{
    delete[] eData;
    delete[] wData;
}


template<class T>
void 
MPIOpenCLStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    if( this->ParticipatingInProgram() )
    {
        // we need to do a halo exchange before our first push of
        // data onto the device
        if( dumpData )
        {
            this->DumpData( ofs, mtx, "before halo exchange" );
        }
        this->DoHaloExchange( mtx );
        if( dumpData )
        {
            this->DumpData( ofs, mtx, "after halo exchange" );
        }

        // allocate memory for halo exchange buffers for non-contiguous sides
        unsigned int haloWidth = this->GetNumberIterationsPerHaloExchange();
        size_t ewDataItemCount = haloWidth * mtx.GetNumRows();

        eData = new T[ewDataItemCount];
        wData = new T[ewDataItemCount];

        // apply the operator
        OpenCLStencil<T>::operator()( mtx, nIters );

        if( dumpData )
        {
            this->DumpData( ofs, mtx, "after all iterations" );
        }
    }
    MPI_Barrier( MPI_COMM_WORLD );
}


template<class T>
void
MPIOpenCLStencil<T>::DoPreIterationWork( cl::Buffer& currBuf,
                                        cl::Buffer& altBuf,
                                        Matrix2D<T>& mtx,
                                        unsigned int iter,
                                        cl::CommandQueue& queue )
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
        std::vector<cl::Event> waitEvents;
        cl::Event nEvent;
        cl::Event sEvent;
        cl::Event eEvent;
        cl::Event wEvent;


        size_t nsDataItemCount = haloWidth * nPaddedCols;
        size_t ewDataItemCount = haloWidth * nRows;
        size_t nsDataSize = nsDataItemCount * sizeof(T);
        size_t ewDataSize = ewDataItemCount * sizeof(T);

        // read current data off device
        // OpenCL 1.0 does not have a strided read, so we need to get
        // the non-contiguous halo sides into contiguous buffers 
        // before reading into our host buffers
        cl::Buffer contigEBuf( queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, ewDataSize );
        cl::Buffer contigWBuf( queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, ewDataSize );
        cl::KernelFunctor copyRectFunc = this->copyRectKernel.bind( queue,
            cl::NDRange( nRows ),
            cl::NullRange );

        if( this->HaveNorthNeighbor() )
        {
            // north data is contiguous - copy directly into matrix
            queue.enqueueReadBuffer( currBuf,
                                        CL_FALSE,    // blocking
                                        haloWidth * nPaddedCols * sizeof(T),          // offset
                                        nsDataSize, // size
                                        flatData + (haloWidth * nPaddedCols),
                                        NULL,
                                        &nEvent );
            waitEvents.push_back( nEvent );
        }

        if( this->HaveSouthNeighbor() )
        {
            // south data is contiguous - copy directly into matrix
            queue.enqueueReadBuffer( currBuf,
                                        CL_FALSE,    // blocking
                                        (nRows - 2 * haloWidth) * nPaddedCols * sizeof(T),    //offset
                                        nsDataSize,
                                        flatData + ((nRows - 2*haloWidth) * nPaddedCols),
                                        NULL,
                                        &sEvent );
            waitEvents.push_back( sEvent );
        }

        if( this->HaveEastNeighbor() )
        {
            // east data is non-contigous - 
            // make it contiguous on the device
            cl::Event ceEvent = copyRectFunc( contigEBuf,   // dest
                            (int)0,          // dest offset
                            (int)haloWidth,  // dest pitch
                            currBuf,    // src
                            (int)(nCols - 2 * haloWidth),          // src offset
                            (int)nPaddedCols,      // src pitch
                            (int)haloWidth,  // width
                            (int)nRows );    // height

            // copy data into contiguous array on host
            std::vector<cl::Event> ceEvents;
            ceEvents.push_back( ceEvent );
            queue.enqueueReadBuffer( contigEBuf,
                                        CL_FALSE,    // blocking
                                        0,          // offset
                                        ewDataSize, // size
                                        eData,
                                        &ceEvents,
                                        &eEvent );
            waitEvents.push_back( eEvent );
        }

        if( this->HaveWestNeighbor() )
        {
            // west data is non-contiguous - 
            // make it contiguous on the device,
            cl::Event cwEvent = copyRectFunc( contigWBuf,   // dest
                            (int)0,          // dest offset
                            (int)haloWidth,  // dest pitch
                            currBuf,    // src
                            (int)haloWidth,          // src offset
                            (int)nPaddedCols,              // src pitch
                            (int)haloWidth,  // width
                            (int)nRows );    // height

            // copy into a contiguous array on the host
            std::vector<cl::Event> cwEvents;
            cwEvents.push_back( cwEvent );
            queue.enqueueReadBuffer( contigWBuf,
                                        CL_FALSE,    // blocking
                                        0,          // offset
                                        ewDataSize, // size
                                        wData,
                                        &cwEvents,
                                        &wEvent );
            waitEvents.push_back( wEvent );
        }


        // wait for all reads from device to complete
        if( !waitEvents.empty() )
        {
            cl::Event::waitForEvents( waitEvents );
            waitEvents.clear();
        }

        // put east/west data into correct position in matrix
        if( this->HaveEastNeighbor() )
        {
            for( unsigned int r = 0; r < nRows; r++ )
            {
                for( unsigned int hc = 0; hc < haloWidth; hc++ )
                {
                    // east side
                    flatData[r*nPaddedCols + 
                                (nCols - 2*haloWidth) + hc] = 
                            eData[r*haloWidth + hc];
                }
            }
        }
        if( this->HaveWestNeighbor() )
        {
            for( unsigned int r = 0; r < nRows; r++ )
            {
                for( unsigned int hc = 0; hc < haloWidth; hc++ )
                {
                    // west side
                    flatData[r*nPaddedCols + 
                                haloWidth + hc] = 
                            wData[r*haloWidth + hc];
                }
            }
        }

        if( dumpData )
        {
            this->DumpData( ofs, mtx, "before halo exchange" );
        }
        this->DoHaloExchange( mtx );
        if( dumpData )
        {
            this->DumpData( ofs, mtx, "after halo exchange" );
        }

        // push updated data back onto device in contiguous buffers
        if( this->HaveNorthNeighbor() )
        {
            queue.enqueueWriteBuffer( currBuf,
                                            CL_FALSE,    // blocking
                                            0,  // offset
                                            nsDataSize, // size
                                            flatData,
                                            NULL,
                                            &nEvent );
            waitEvents.push_back( nEvent );
        }
        if( this->HaveSouthNeighbor() )
        {
            queue.enqueueWriteBuffer( currBuf,
                                            CL_FALSE,    // blocking
                                            (nRows - haloWidth) * nPaddedCols * sizeof(T),    //offset
                                            nsDataSize,
                                            flatData + ((nRows - haloWidth) * nPaddedCols),
                                            NULL,
                                            &sEvent );
            waitEvents.push_back( sEvent );
        }

        if( this->HaveEastNeighbor() )
        {
            // copy from mtx to contiguous buffers
            for( unsigned int r = 0; r < nRows; r++ )
            {
                for( unsigned int hc = 0; hc < haloWidth; hc++ )
                {
                    // east side
                    eData[r*haloWidth + hc] = 
                        flatData[r*nPaddedCols + (nCols - haloWidth) + hc];
                }
            }

            // push up to device
            queue.enqueueWriteBuffer( contigEBuf,
                                            CL_FALSE,    // blocking
                                            0,          // offset
                                            ewDataSize, // size
                                            eData,
                                            NULL,
                                            &eEvent );
            waitEvents.push_back( eEvent );
        }

        if( this->HaveWestNeighbor() )
        {
            // copy from mtx to contiguous buffers
            for( unsigned int r = 0; r < nRows; r++ )
            {
                for( unsigned int hc = 0; hc < haloWidth; hc++ )
                {
                    // west side
                    wData[r*haloWidth + hc] = flatData[r*nPaddedCols + hc];
                }
            }

            // push up to device
            queue.enqueueWriteBuffer( contigWBuf,
                                            CL_FALSE,    // blocking
                                            0,          // offset
                                            ewDataSize, // size
                                            wData,
                                            NULL,
                                            &wEvent );
            waitEvents.push_back( wEvent );
        }
        if( !waitEvents.empty() )
        {
            cl::Event::waitForEvents( waitEvents );
            waitEvents.clear();
        }

        if( this->HaveEastNeighbor() )
        {
            eEvent = copyRectFunc( currBuf,      // dest
                                (int)(nCols - haloWidth),    // dest offset
                                (int)nPaddedCols,      // dest pitch
                                contigEBuf,   // src
                                (int)0,          // src offset
                                (int)haloWidth,  // src pitch
                                (int)haloWidth,  // width
                                (int)nRows );    // height
            waitEvents.push_back( eEvent );
        }
        if( this->HaveWestNeighbor() )
        {
            wEvent = copyRectFunc( currBuf,      // dest
                                (int)0,  // dest offset
                                (int)nPaddedCols,      // dest pitch
                                contigWBuf, // src
                                (int)0,          // src offset
                                (int)haloWidth,  // src pitch
                                (int)haloWidth,  // width
                                (int)nRows );    // height
            waitEvents.push_back( wEvent );
        }
        if( !waitEvents.empty() )
        {
            cl::Event::waitForEvents( waitEvents );
            waitEvents.clear();
        }

        // we may have changed the local halo values on the device
        // we need to update the local 1-wide halo onto the alt buffer
        // note we only need to update the 1-wide halo here, even if
        // our real halo width is larger
        size_t rowExtent = mtx.GetNumPaddedColumns() * sizeof(T);
        if( this->HaveNorthNeighbor() )
        {
            queue.enqueueCopyBuffer( currBuf,
                                        altBuf,
                                        0,
                                        0,
                                        rowExtent,
                                        NULL,
                                        &nEvent );
            waitEvents.push_back( nEvent );
        }
        if( this->HaveSouthNeighbor() )
        {
            queue.enqueueCopyBuffer( currBuf,
                                        altBuf,
                                        (mtx.GetNumRows() - 1) * rowExtent,
                                        (mtx.GetNumRows() - 1) * rowExtent,
                                        rowExtent,
                                        NULL,
                                        &sEvent );
            waitEvents.push_back( sEvent );
        }
        if( this->HaveEastNeighbor() )
        {
            eEvent = copyRectFunc( altBuf,         // dest
                            (int)(mtx.GetNumColumns() - 1),              // dest offset
                            (int)mtx.GetNumPaddedColumns(),    // dest pitch
                            currBuf,      // src
                            (int)(mtx.GetNumColumns() - 1),              // src offset
                            (int)mtx.GetNumPaddedColumns(),    // src pitch
                            1,              // width
                            (int)mtx.GetNumRows() );        // height
            waitEvents.push_back( eEvent );
        }
        if( this->HaveWestNeighbor() )
        {
            wEvent = copyRectFunc( altBuf,         // dest
                            0,              // dest offset
                            (int)mtx.GetNumPaddedColumns(),    // dest pitch
                            currBuf,      // src
                            0,              // src offset
                            (int)mtx.GetNumPaddedColumns(),    // src pitch
                            1,              // width
                            (int)mtx.GetNumRows() );        // height
            waitEvents.push_back( wEvent );
        }
        if( !waitEvents.empty() )
        {
            cl::Event::waitForEvents( waitEvents );
            waitEvents.clear();
        }
    }
}

