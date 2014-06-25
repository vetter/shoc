#include "mpi.h"
#include <sstream>
#include <iomanip>
#include <cassert>
#include "MPIOpenCLStencil.h"
#include "support.h"



template<class T>
MPIOpenCLStencil<T>::MPIOpenCLStencil( T _wCenter,
                                    T _wCardinal,
                                    T _wDiagonal,
                                    size_t _lRows,
                                    size_t _lCols,
                                    size_t _mpiGridRows,
                                    size_t _mpiGridCols,
                                    unsigned int _nItersPerHaloExchange,
                                    cl_device_id _dev,
                                    cl_context _ctx,
                                    cl_command_queue _queue,
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
MPIOpenCLStencil<T>::DoPreIterationWork( cl_mem currBuf,
                                        cl_mem altBuf,
                                        Matrix2D<T>& mtx,
                                        unsigned int iter,
                                        cl_command_queue queue )
{
    cl_int clErr;

    // do the halo exchange at desired frequency
    // note that we *do not* do the halo exchange here before the
    // first iteration, because we did it already (before we first
    // pushed the data onto the device) in our operator() method.
    unsigned int haloWidth = this->GetNumberIterationsPerHaloExchange();
    if( (iter > 0) && (iter % haloWidth) == 0 )
    {
        size_t nRows = mtx.GetNumRows();
        size_t nCols = mtx.GetNumColumns();
        unsigned int nPaddedCols = mtx.GetNumPaddedColumns();
        T* flatData = mtx.GetFlatData();
        std::vector<cl_event> waitEvents;
        cl_event nEvent;
        cl_event sEvent;
        cl_event eEvent;
        cl_event wEvent;


        size_t nsDataItemCount = haloWidth * nPaddedCols;
        size_t ewDataItemCount = haloWidth * nRows;
        size_t nsDataSize = nsDataItemCount * sizeof(T);
        size_t ewDataSize = ewDataItemCount * sizeof(T);

        // read current data off device
        // OpenCL 1.0 does not have a strided read, so we need to get
        // the non-contiguous halo sides into contiguous buffers
        // before reading into our host buffers
        cl_mem contigEBuf = clCreateBuffer( this->GetContext(),
                                            CL_MEM_READ_WRITE,
                                            ewDataSize,
                                            NULL,
                                            &clErr );
        CL_CHECK_ERROR(clErr);
        cl_mem contigWBuf = clCreateBuffer( this->GetContext(),
                                            CL_MEM_READ_WRITE,
                                            ewDataSize,
                                            NULL,
                                            &clErr );
        CL_CHECK_ERROR(clErr);

        if( this->HaveNorthNeighbor() )
        {
            // north data is contiguous - copy directly into matrix
            clErr = clEnqueueReadBuffer(queue,
                                        currBuf,
                                        CL_FALSE,   // is it blocking
                                        haloWidth * nPaddedCols * sizeof(T),    // offset
                                        nsDataSize, // size
                                        flatData + (haloWidth * nPaddedCols),   // dest
                                        0,      // num events to wait on
                                        NULL,   // events to wait on
                                        &nEvent);   // completion event
            CL_CHECK_ERROR(clErr);
            waitEvents.push_back( nEvent );
        }

        if( this->HaveSouthNeighbor() )
        {
            // south data is contiguous - copy directly into matrix
            clErr = clEnqueueReadBuffer(queue,
                                        currBuf,
                                        CL_FALSE,    // is it blocking?
                                        (nRows - 2 * haloWidth) * nPaddedCols * sizeof(T),    //offset
                                        nsDataSize,
                                        flatData + ((nRows - 2*haloWidth) * nPaddedCols),
                                        0,      // num events to wait on
                                        NULL,   // events to wait on
                                        &sEvent);  // completion event
            CL_CHECK_ERROR(clErr);
            waitEvents.push_back( sEvent );
        }

        if( this->HaveEastNeighbor() )
        {
            // east data is non-contigous -
            // make it contiguous on the device
            this->SetCopyRectKernelArgs( contigEBuf, // dest
                                    (int)0, // dest offset
                                    (int)haloWidth, // dest pitch
                                    currBuf, // src
                                    (int)(nCols - 2 * haloWidth), // src offset
                                    (int)nPaddedCols, // src pitch
                                    (int)haloWidth,  // width
                                    (int)nRows ); // height

            cl_event ceEvent;
            clErr = clEnqueueNDRangeKernel(queue,
                                            this->copyRectKernel,
                                            1,  // number of work dimensions
                                            NULL,   // global work offset - use all 0s
                                            &nRows, // global work size
                                            NULL,   // local work size - impl defined
                                            0,  // num events to wait on
                                            NULL,   // events to wait on
                                            &ceEvent);  // completion event
            CL_CHECK_ERROR(clErr);

            // copy data into contiguous array on host
            std::vector<cl_event> ceEvents;
            ceEvents.push_back( ceEvent );
            clErr = clEnqueueReadBuffer(queue,
                                        contigEBuf,
                                        CL_FALSE,   // is it blocking?
                                        0,          // offset
                                        ewDataSize, // size
                                        eData,
                                        ceEvents.size(),    // num events to wait on
                                        ceEvents.empty() ? NULL : &ceEvents.front(),   // events to wait on
                                        &eEvent);   // completion event
            CL_CHECK_ERROR(clErr);
            waitEvents.push_back( eEvent );
        }

        if( this->HaveWestNeighbor() )
        {
            // west data is non-contiguous -
            // make it contiguous on the device,
            this->SetCopyRectKernelArgs( contigWBuf,  // dest
                                    (int)0, // dest offset
                                    (int)haloWidth, // dest pitch
                                    currBuf, // src
                                    (int)haloWidth, // src offset
                                    (int)nPaddedCols, // src pitch
                                    (int)haloWidth, // width
                                    (int)nRows ); // height

            cl_event cwEvent;
            clErr = clEnqueueNDRangeKernel(queue,
                                            this->copyRectKernel,
                                            1,  // number of work dimensions
                                            NULL,   // global work offset - use all 0s
                                            &nRows, // global work size
                                            NULL,   // local work size - impl defined
                                            0,      // num events to wait on
                                            NULL,   // events to wait on
                                            &cwEvent);  // completion event
            CL_CHECK_ERROR(clErr);

            // copy into a contiguous array on the host
            std::vector<cl_event> cwEvents;
            cwEvents.push_back( cwEvent );
            clErr = clEnqueueReadBuffer(queue,
                                        contigWBuf,
                                        CL_FALSE,   // is it blocking?
                                        0,          // offset
                                        ewDataSize, // size
                                        wData,      // host location
                                        cwEvents.size(),    // num events to wait on
                                        cwEvents.empty() ? NULL : &cwEvents.front(),   // events to wait on
                                        &wEvent );  // completion event

            CL_CHECK_ERROR(clErr);                                            
            waitEvents.push_back( wEvent );
        }


        // wait for all reads from device to complete
        if( !waitEvents.empty() )
        {
            clErr = clWaitForEvents( waitEvents.size(), &waitEvents.front() );
            CL_CHECK_ERROR(clErr);
            this->ClearWaitEvents( waitEvents );
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
            clErr = clEnqueueWriteBuffer(queue,
                                            currBuf,
                                            CL_FALSE,   // is it blocking?
                                            0,          // offset
                                            nsDataSize, // size
                                            flatData,   // host location
                                            0,  // num events to wait on
                                            NULL,   // events to wait on
                                            &nEvent);   // completion event
            CL_CHECK_ERROR(clErr);
            waitEvents.push_back( nEvent );
        }
        if( this->HaveSouthNeighbor() )
        {
            clErr = clEnqueueWriteBuffer(queue,
                                            currBuf,
                                            CL_FALSE,   // is it blocking?
                                            (nRows - haloWidth) * nPaddedCols * sizeof(T),    //offset
                                            nsDataSize, // size
                                            flatData + ((nRows - haloWidth) * nPaddedCols), // host location
                                            0,      // num events to wait on
                                            NULL,   // events to wait on
                                            &sEvent);   // completion event
            CL_CHECK_ERROR(clErr);
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
            clErr = clEnqueueWriteBuffer(queue,
                                        contigEBuf,
                                        CL_FALSE,    // is it blocking?
                                        0,          // offset
                                        ewDataSize, // size
                                        eData,  // host location
                                        0,      // num events to wait on
                                        NULL,   // events to wait on
                                        &eEvent );  // completion event
            CL_CHECK_ERROR(clErr);
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
            clErr = clEnqueueWriteBuffer(queue,
                                        contigWBuf,
                                        CL_FALSE,    // is it blocking?
                                        0,          // offset
                                        ewDataSize, // size
                                        wData,      // host location
                                        0,          // num events to wait on
                                        NULL,       // events to wait on
                                        &wEvent );  // completion event

            CL_CHECK_ERROR(clErr);
            waitEvents.push_back( wEvent );
        }
        if( !waitEvents.empty() )
        {
            clErr = clWaitForEvents( waitEvents.size(), &waitEvents.front() );
            CL_CHECK_ERROR(clErr);
            this->ClearWaitEvents( waitEvents );
        }

        if( this->HaveEastNeighbor() )
        {
            this->SetCopyRectKernelArgs( currBuf, // dest
                                    (int)(nCols - haloWidth), // dest offset
                                    (int)nPaddedCols, // dest pitch
                                    contigEBuf, // src
                                    (int)0, // src offset
                                    (int)haloWidth, // src pitch
                                    (int)haloWidth, // width
                                    (int)nRows ); // height

            clErr = clEnqueueNDRangeKernel(queue,
                                        this->copyRectKernel,
                                        1,  // num work dimensions
                                        NULL,   // global work offset - use all 0s
                                        &nRows, // global work size
                                        NULL,   // local work size
                                        0,      // number of events to wait on
                                        NULL,   // events to wait on
                                        &eEvent);   // completion event
            CL_CHECK_ERROR(clErr);
            waitEvents.push_back( eEvent );
        }
        if( this->HaveWestNeighbor() )
        {
            this->SetCopyRectKernelArgs( currBuf, // dest
                                    (int)0, // dest offset
                                    (int)nPaddedCols, // dest pitch
                                    contigWBuf, // src
                                    (int)0, // src offset
                                    (int)haloWidth, // src pitch
                                    (int)haloWidth, // width
                                    (int)nRows ); // height

            clErr = clEnqueueNDRangeKernel(queue,
                                        this->copyRectKernel,
                                        1,  // num work dimensions
                                        NULL,   // global work offset - use all 0s
                                        &nRows, // global work size
                                        NULL,   // local work size
                                        0,      // number of events to wait on
                                        NULL,   // events to wait on
                                        &wEvent);   // completion event
            CL_CHECK_ERROR(clErr);
            waitEvents.push_back( wEvent );
        }
        if( !waitEvents.empty() )
        {
            clErr = clWaitForEvents( waitEvents.size(), &waitEvents.front() );
            CL_CHECK_ERROR(clErr);
            this->ClearWaitEvents( waitEvents );
        }

        // we may have changed the local halo values on the device
        // we need to update the local 1-wide halo onto the alt buffer
        // note we only need to update the 1-wide halo here, even if
        // our real halo width is larger
        size_t rowExtent = mtx.GetNumPaddedColumns() * sizeof(T);
        if( this->HaveNorthNeighbor() )
        {
            clErr = clEnqueueCopyBuffer(queue,
                                        currBuf,    // src buffer
                                        altBuf,     // dest buffer
                                        0,          // src offset
                                        0,          // dest offset
                                        rowExtent,  // nbytes to copy
                                        0,          // num events to wait on
                                        NULL,       // events to wait on
                                        &nEvent);   // completion event
            CL_CHECK_ERROR(clErr);
            waitEvents.push_back( nEvent );
        }
        if( this->HaveSouthNeighbor() )
        {
            clErr = clEnqueueCopyBuffer(queue,
                                        currBuf,    // src buffer
                                        altBuf,     // dest buffer
                                        (mtx.GetNumRows() - 1) * rowExtent, // src offset
                                        (mtx.GetNumRows() - 1) * rowExtent, // dest offset
                                        rowExtent,  // nbytes to copy
                                        0,          // num events to wait on
                                        NULL,       // events to wait on
                                        &sEvent );  // completion event
            CL_CHECK_ERROR(clErr);
            waitEvents.push_back( sEvent );
        }
        if( this->HaveEastNeighbor() )
        {
            this->SetCopyRectKernelArgs( altBuf, // dest
                                    (int)(mtx.GetNumColumns() - 1), // dest offset
                                    (int)mtx.GetNumPaddedColumns(), // dest pitch
                                    currBuf, // src
                                    (int)(mtx.GetNumColumns() - 1), // src offset
                                    (int)mtx.GetNumPaddedColumns(), // src pitch
                                    1, // width
                                    (int)mtx.GetNumRows() ); // height

            clErr = clEnqueueNDRangeKernel(queue,
                                            this->copyRectKernel,
                                            1,  // num work dims
                                            NULL,   // global work offset - use all 0s
                                            &nRows, // global work size
                                            NULL,   // local work size - impl defined
                                            0,      // num events to wait on
                                            NULL,   // events to wait on
                                            &eEvent);   // completion event
            CL_CHECK_ERROR(clErr);
            waitEvents.push_back( eEvent );
        }
        if( this->HaveWestNeighbor() )
        {
            this->SetCopyRectKernelArgs( altBuf, // dest
                                    0, // dest offset
                                    (int)mtx.GetNumPaddedColumns(), // dest pitch
                                    currBuf, // src
                                    0, // src offset
                                    (int)mtx.GetNumPaddedColumns(), // src pitch
                                    1, // width
                                    (int)mtx.GetNumRows() ); // height

            clErr = clEnqueueNDRangeKernel(queue,
                                            this->copyRectKernel,
                                            1,  // num work dims
                                            NULL,   // global work offset - use all 0s
                                            &nRows, // global work size
                                            NULL,   // local work size
                                            0,      // num events to wait on
                                            NULL,   // events to wait on
                                            &wEvent);   // completion event
            CL_CHECK_ERROR(clErr);
            waitEvents.push_back( wEvent );
        }
        if( !waitEvents.empty() )
        {
            clErr = clWaitForEvents( waitEvents.size(), &waitEvents.front() );
            CL_CHECK_ERROR(clErr);
            this->ClearWaitEvents( waitEvents );
        }

        clErr = clReleaseMemObject( contigEBuf );
        CL_CHECK_ERROR(clErr);
        clErr = clReleaseMemObject( contigWBuf );
        CL_CHECK_ERROR(clErr);
    }
}

