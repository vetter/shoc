#include <cassert>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include "shoc_compat_cas.h"
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "Stencil.h"
#include "OpenCLStencil.h"
#include "InvalidArgValue.h"
#include "support.h"

extern const char *cl_source_stencil2d;

template<class T>
OpenCLStencil<T>::OpenCLStencil( T wCenter,
                    T wCardinal,
                    T wDiagonal,
                    size_t _lRows,
                    size_t _lCols,
                    cl::Device& _dev,
                    cl::Context& _ctx,
                    cl::CommandQueue& _queue )
  : Stencil<T>( wCenter, wCardinal, wDiagonal ),
    lRows( _lRows ),
    lCols( _lCols ),
    device( _dev ),
    context( _ctx ),
    queue( _queue )
{
    // determine our value type (as a string)
    std::string precision;
#if READY
    // use RTTI to determine type? 
#else
    // avoid using RTTI unless we have to - 
    // instead, use an ad hoc approach to determining whether
    // we are working with floats or doubles
    if( sizeof(T) == sizeof(float) )
    {
        precision = "SINGLE_PRECISION";
    }
    else if( sizeof(T) == sizeof(double) )
    {
        if( checkExtension( device(), "cl_khr_fp64" ) )
        {
            precision = "K_DOUBLE_PRECISION";
        }
        else if( checkExtension( device(), "cl_amd_fp64" ) )
        {
            precision = "AMD_DOUBLE_PRECISION";
        }
        else
        {
            throw InvalidArgValue( "Double precision not supported by chosen device" );
        }
    }
    else
    {
        throw InvalidArgValue( "template type T has unrecognized size (must be float or double)" );
    }
#endif // READY

    // associate our OpenCL kernel source with a Program
    cl::Program::Sources source( 1,
        std::make_pair(cl_source_stencil2d,strlen(cl_source_stencil2d)) );
    cl::Program prog( context, source );
    std::ostringstream buildOptions;
    buildOptions << "-DLROWS=" << lRows 
                << " -DLCOLS=" << lCols
                << " -D" << precision;
    try
    {
        prog.build( context.getInfo<CL_CONTEXT_DEVICES>(), buildOptions.str().c_str() );
    }
    catch( ... )
    {
        // the build failed - why?
        std::cerr << "Failed to build OpenCL program.  Build log:\n"
            << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>( device )
            << std::endl;

        throw;
    }

    kernel = cl::Kernel( prog, "StencilKernel" );
    copyRectKernel = cl::Kernel( prog, "CopyRect" );
}





template<class T>
void
OpenCLStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    std::vector<cl::Event> waitEvents;

    // determine local workgroup size
    // assumes mtx has been padded with halo of width >= 1
    //
    // Since each GPU thread is responsible for a strip of data
    // from the original, our index space is scaled smaller 
    // in one dimension relative to the actual data
    assert( ((mtx.GetNumRows() - 2) % lRows) == 0 );

    // create buffers for our data on the device
    cl::Buffer dataBuf1( context, CL_MEM_READ_WRITE, mtx.GetDataSize() );
    cl::Buffer dataBuf2( context, CL_MEM_READ_WRITE, mtx.GetDataSize() );
    cl::Buffer* currData = &dataBuf1;
    cl::Buffer* newData = &dataBuf2;

    // copy the initial matrix values onto the device
    cl::Event writeEvt;
    queue.enqueueWriteBuffer( *currData,
                                CL_FALSE,    // blocking
                                0,          // offset
                                mtx.GetDataSize(),
                                mtx.GetFlatData(),
                                NULL,       // wait on no earlier events
                                &writeEvt );    // save completion event
    waitEvents.push_back( writeEvt );

    // copy the halo from the initial buffer into the second buffer
    // Note: when doing local iterations, these values do not change
    // but they can change in the MPI version after an inter-process
    // halo exchange.
    //
    // copy the sides with contiguous data, ensuring push of initial data 
    // to device has completed
    size_t rowExtent = mtx.GetNumPaddedColumns() * sizeof(T);
    queue.enqueueCopyBuffer( *currData,
                                *newData,
                                0,
                                0,
                                rowExtent,
                                &waitEvents,
                                NULL );
    queue.enqueueCopyBuffer( *currData,
                                *newData,
                                (mtx.GetNumRows() - 1) * rowExtent,
                                (mtx.GetNumRows() - 1) * rowExtent,
                                rowExtent,
                                NULL,
                                NULL );

    // copy the non-contiguous data
    // NOTE: OpenCL 1.1 provides a function clEnqueueCopyBufferRect that 
    // seems like it would be useful here.  For OpenCL 1.0 compatibility,
    // we use a custom kernel that copies the non-contiguous data.
    waitEvents.clear();
    copyRectKernel.setArg( 0, *newData ); // dest
    copyRectKernel.setArg( 1, 0 ); // dest offset
    copyRectKernel.setArg( 2, (int)mtx.GetNumPaddedColumns() ); // dest pitch
    copyRectKernel.setArg( 3, *currData ); // src
    copyRectKernel.setArg( 4, 0 ); // src offset
    copyRectKernel.setArg( 5, (int)mtx.GetNumPaddedColumns() ); // src pitch
    copyRectKernel.setArg( 6, 1 ); // width
    copyRectKernel.setArg( 7, (int)mtx.GetNumRows() ); // height

    cl::Event cwEvent;
    queue.enqueueNDRangeKernel( copyRectKernel,
        cl::NullRange,
        cl::NDRange( mtx.GetNumRows() ),
        cl::NullRange,
        NULL,
        &cwEvent );
    waitEvents.push_back( cwEvent );

    copyRectKernel.setArg( 0, *newData ); // dest
    copyRectKernel.setArg( 1, (int)(mtx.GetNumColumns() - 1) ); // dest offset
    copyRectKernel.setArg( 2, (int)mtx.GetNumPaddedColumns() ); // dest pitch
    copyRectKernel.setArg( 3, *currData ); // src
    copyRectKernel.setArg( 4, (int)(mtx.GetNumColumns() - 1) ); // src offset
    copyRectKernel.setArg( 5, (int)mtx.GetNumPaddedColumns() ); // src pitch
    copyRectKernel.setArg( 6, 1 ); // width
    copyRectKernel.setArg( 7, (int)mtx.GetNumRows() ); // height
    cl::Event ceEvent;
    queue.enqueueNDRangeKernel( copyRectKernel,
        cl::NullRange,
        cl::NDRange( mtx.GetNumRows() ),
        cl::NullRange,
        NULL,
        &ceEvent );
    waitEvents.push_back( ceEvent );

    cl::Event::waitForEvents( waitEvents );

    // do the stencil iterations
    waitEvents.clear();
    for( int iter = 0; iter < nIters; iter++ )
    {
        this->DoPreIterationWork( *currData,
                                    *newData,
                                    mtx,
                                    iter,
                                    queue );

        // we would like to use event dependency list as specified 
        // in the OpenCL spec and in the C++ binding API, 
        // allowing the OpenCL runtime to manage dependencies on these
        // kernel invocations, but the C++ binding API 
        // silently (!) ignores the events vector.
        //
        // Instead, we go back all the way to host code to wait
        // for one iteration to finish before enqueuing the next.
        // We have one optimization for the final iteration - 
        // in that case we enqueue the read buffer command and
        // make it dependent on the completion of the last iteration.
        //
        size_t localDataSize = (lRows + 2) * (lCols + 2) * sizeof(T);

        // We would like to use a C++ functor approach, but
        // the KernelFunctor from the earlier OpenCL C++ bindings 
        // has disappeared and anecdotal evidence suggests that
        // the make_kernel approach in the OpenCL 1.2-related C++ bindings
        // might not be stable a stable API.
        // So we stick with the verbose setArg/enqueueNDRangeKernel approach.
        kernel.setArg( 0, *currData );
        kernel.setArg( 1, *newData );
        kernel.setArg( 2, (cl_int)(mtx.GetPad()) );
        kernel.setArg( 3, this->wCenter );
        kernel.setArg( 4, this->wCardinal );
        kernel.setArg( 5, this->wDiagonal );
        kernel.setArg( 6, cl::__local( localDataSize ) );

        cl::Event evt;
        queue.enqueueNDRangeKernel( kernel,
            cl::NullRange,
            cl::NDRange( (mtx.GetNumRows() - 2) / lRows, mtx.GetNumColumns() - 2 ),
            cl::NDRange( 1, lCols ), 
            NULL,
            &evt );

        if( iter == nIters-1 )
        {
            // last iteration - put the event in the dependency list
            // to be used by the read buffer command
            waitEvents.push_back( evt );
        }
        else
        {
            // Not the last iteration - more iterations to follow.
            // wait for this iteration to complete.
            evt.wait();
        }

        // switch to put new data into other buffer
        if( currData == &dataBuf1 )
        {
            currData = &dataBuf2;            
            newData = &dataBuf1;
        }
        else
        {
            currData = &dataBuf1;
            newData = &dataBuf2;
        }
    }

    // read final data off the device
    queue.enqueueReadBuffer( *currData,
                                CL_TRUE,    // blocking
                                0,          // offset
                                mtx.GetDataSize(),
                                mtx.GetFlatData(),
                                &waitEvents );
}


template<class T>
void
OpenCLStencil<T>::DoPreIterationWork( cl::Buffer& buf,
                                    cl::Buffer& altBuf,
                                    Matrix2D<T>& mtx,
                                    unsigned int iter,
                                    cl::CommandQueue& queue )
{
    // in single process version, nothing for us to do
}

