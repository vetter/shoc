#include <cassert>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
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
                    cl_device_id _dev,
                    cl_context _ctx,
                    cl_command_queue _queue )
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
        if( checkExtension( device, "cl_khr_fp64" ) )
        {
            precision = "K_DOUBLE_PRECISION";
        }
        else if( checkExtension( device, "cl_amd_fp64" ) )
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
    cl_int clErr;
    cl_program prog = clCreateProgramWithSource( context,
                                                1,
                                                &cl_source_stencil2d,
                                                NULL,
                                                &clErr );
    CL_CHECK_ERROR(clErr);

    std::ostringstream buildOptions;
    buildOptions << "-DLROWS=" << lRows
                << " -DLCOLS=" << lCols
                << " -D" << precision;

    try
    {
        clErr = clBuildProgram( prog,
                                1,
                                &device,
                                buildOptions.str().c_str(),
                                NULL,
                                NULL );
        CL_CHECK_ERROR(clErr);
    }
    catch( ... )
    {
        // the build failed - why?
        std::string msg = "build failed";
        size_t nBytesNeeded = 0;
        clErr = clGetProgramBuildInfo( prog,
                                        device,
                                        CL_PROGRAM_BUILD_LOG,
                                        0,
                                        NULL,
                                        &nBytesNeeded );
        if( clErr == CL_SUCCESS )
        {
            char* buildLog = new char[nBytesNeeded+1];
            clErr = clGetProgramBuildInfo( prog,
                                            device,
                                            CL_PROGRAM_BUILD_LOG,
                                            nBytesNeeded+1,
                                            buildLog,
                                            NULL );
            if( clErr == CL_SUCCESS )
            {
                msg = buildLog;
            }
            delete[] buildLog;
        }

        std::cerr << "Failed to build OpenCL program.  Build log:\n"
            << msg
            << std::endl;

        throw;
    }

    kernel = clCreateKernel( prog, "StencilKernel", &clErr );
    CL_CHECK_ERROR(clErr);
    copyRectKernel = clCreateKernel( prog, "CopyRect", &clErr );
    CL_CHECK_ERROR(clErr);
}



template<class T>
void
OpenCLStencil<T>::SetCopyRectKernelArgs( cl_mem dest,
                                            int destOffset,
                                            int destPitch,
                                            cl_mem src,
                                            int srcOffset,
                                            int srcPitch,
                                            int width,
                                            int height )
{
    int clErr;

    clErr = clSetKernelArg(copyRectKernel, 0, sizeof(cl_mem), &dest );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(copyRectKernel, 1, sizeof(int), &destOffset );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(copyRectKernel, 2, sizeof(int), &destPitch );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(copyRectKernel, 3, sizeof(cl_mem), &src );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(copyRectKernel, 4, sizeof(int), &srcOffset );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(copyRectKernel, 5, sizeof(int), &srcPitch );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(copyRectKernel, 6, sizeof(int), &width );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(copyRectKernel, 7, sizeof(int), &height );
    CL_CHECK_ERROR(clErr);
}


template<class T>
void
OpenCLStencil<T>::SetStencilKernelArgs( cl_mem currData,
                                        cl_mem newData,
                                        int alignment,
                                        T wCenter,
                                        T wCardinal,
                                        T wDiagonal,
                                        size_t localDataSize )
{
    cl_int clErr;

    clErr = clSetKernelArg(kernel, 0, sizeof(cl_mem), &currData );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(kernel, 1, sizeof(cl_mem), &newData );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(kernel, 2, sizeof(int), &alignment );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(kernel, 3, sizeof(T), &wCenter );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(kernel, 4, sizeof(T), &wCardinal );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(kernel, 5, sizeof(T), &wDiagonal );
    CL_CHECK_ERROR(clErr);

    clErr = clSetKernelArg(kernel, 6, localDataSize, NULL );
    CL_CHECK_ERROR(clErr);
}


template<class T>
void
OpenCLStencil<T>::ClearWaitEvents( std::vector<cl_event>& waitEvents )
{
    for( std::vector<cl_event>::iterator iter = waitEvents.begin();
            iter != waitEvents.end();
            ++iter )
    {
        clReleaseEvent( *iter );
    }
    waitEvents.clear();
}



template<class T>
void
OpenCLStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    std::vector<cl_event> waitEvents;

    // determine local workgroup size
    // assumes mtx has been padded with halo of width >= 1
    //
    // Since each GPU thread is responsible for a strip of data
    // from the original, our index space is scaled smaller
    // in one dimension relative to the actual data
    assert( ((mtx.GetNumRows() - 2) % lRows) == 0 );

    // create buffers for our data on the device
    cl_int clErr = 0;
    cl_mem dataBuf1 = clCreateBuffer( context,
                            CL_MEM_READ_WRITE,
                            mtx.GetDataSize(),
                            NULL,
                            &clErr );
    CL_CHECK_ERROR(clErr);
    cl_mem dataBuf2 = clCreateBuffer( context,
                            CL_MEM_READ_WRITE,
                            mtx.GetDataSize(),
                            NULL,
                            &clErr );
    CL_CHECK_ERROR(clErr);
    cl_mem* currData = &dataBuf1;
    cl_mem* newData = &dataBuf2;

    // copy the initial matrix values onto the device
    cl_event writeEvt;
    clErr = clEnqueueWriteBuffer( queue,
                                    *currData,
                                    CL_FALSE,   //blocking
                                    0,          // offset
                                    mtx.GetDataSize(),  // cb
                                    mtx.GetFlatData(),
                                    0,          // wait on no earlier events
                                    NULL,
                                    &writeEvt );    // save completion event
    CL_CHECK_ERROR(clErr);
    waitEvents.push_back( writeEvt );


    // copy the halo from the initial buffer into the second buffer
    // Note: when doing local iterations, these values do not change
    // but they can change in the MPI version after an inter-process
    // halo exchange.
    //
    // copy the sides with contiguous data, ensuring push of initial data
    // to device has completed
    size_t rowExtent = mtx.GetNumPaddedColumns() * sizeof(T);
    clErr = clEnqueueCopyBuffer( queue,
                                    *currData,
                                    *newData,
                                    0,
                                    0,
                                    rowExtent,
                                    waitEvents.size(),
                                    waitEvents.empty() ? NULL : &waitEvents.front(),
                                    NULL );
    CL_CHECK_ERROR(clErr);
    ClearWaitEvents( waitEvents );

    clErr = clEnqueueCopyBuffer( queue,
                                    *currData,
                                    *newData,
                                    (mtx.GetNumRows() - 1) * rowExtent,
                                    (mtx.GetNumRows() - 1) * rowExtent,
                                    rowExtent,
                                    0,
                                    NULL,
                                    NULL );

    // copy the non-contiguous data
    // NOTE: OpenCL 1.1 provides a function clEnqueueCopyBufferRect that
    // seems like it would be useful here.  For OpenCL 1.0 compatibility,
    // we use a custom kernel that copies the non-contiguous data.
    SetCopyRectKernelArgs( *newData,  // dest
                            0,  // dest offset
                            (int)mtx.GetNumPaddedColumns(),  // dest pitch
                            *currData,  // src
                            0,  // src offset
                            (int)mtx.GetNumPaddedColumns(),  // src pitch
                            1,  // width
                            (int)mtx.GetNumRows() );  // height

    cl_event cwEvent;
    size_t global_work_size = mtx.GetNumRows();
    clErr = clEnqueueNDRangeKernel( queue,
                                    copyRectKernel,
                                    1,      // work dimensions
                                    NULL,   // global work offset - use all 0s
                                    &global_work_size,  // global work size
                                    NULL,   // local work size - impl defined
                                    0,      // number of events to wait on
                                    NULL,   // events to wait on
                                    &cwEvent ); // completion event
    CL_CHECK_ERROR(clErr);
    waitEvents.push_back( cwEvent );

    SetCopyRectKernelArgs( *newData,  // dest
                            (int)(mtx.GetNumColumns() - 1),  // dest offset
                            (int)mtx.GetNumPaddedColumns(),  // dest pitch
                            *currData,  // src
                            (int)(mtx.GetNumColumns() - 1),  // src offset
                            (int)mtx.GetNumPaddedColumns(),  // src pitch
                            1,  // width
                            (int)mtx.GetNumRows());  // height

    cl_event ceEvent;
    clErr = clEnqueueNDRangeKernel( queue,
                                    copyRectKernel,
                                    1,      // work dimensions
                                    NULL,   // global work offset - use all 0s
                                    &global_work_size,  // global work size
                                    NULL,   // local work size - impl defined
                                    0,      // number of events to wait on
                                    NULL,   // events to wait on
                                    &ceEvent ); // completion event
    CL_CHECK_ERROR(clErr);
    waitEvents.push_back( ceEvent );

    clErr = clWaitForEvents( waitEvents.size(),
                        waitEvents.empty() ? NULL : &waitEvents.front() );
    CL_CHECK_ERROR(clErr);
    ClearWaitEvents( waitEvents );


    // do the stencil iterations
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

        SetStencilKernelArgs( *currData,
                                *newData,
                                (cl_int)(mtx.GetPad()),
                                this->wCenter,
                                this->wCardinal,
                                this->wDiagonal,
                                localDataSize );

        cl_event mainKernelEvt;
        std::vector<size_t> gWorkDims(2, 0);
        gWorkDims[0] = (mtx.GetNumRows() - 2) / lRows;
        gWorkDims[1] = mtx.GetNumColumns() - 2;
        std::vector<size_t> lWorkDims(2, 0);
        lWorkDims[0] = 1;
        lWorkDims[1] = lCols;
        clErr = clEnqueueNDRangeKernel( queue,
                                kernel,
                                gWorkDims.size(),   // number of work dimensions
                                NULL,               // global work offset - use all 0s
                                &gWorkDims.front(),  // global work dimensions
                                &lWorkDims.front(),  // local work dimensions
                                0,      // number of events to wait on
                                NULL,   // events to wait on
                                &mainKernelEvt ); // completion event
        CL_CHECK_ERROR(clErr);

        if( iter == nIters-1 )
        {
            // last iteration - put the event in the dependency list
            // to be used by the read buffer command
            waitEvents.push_back( mainKernelEvt );
        }
        else
        {
            // Not the last iteration - more iterations to follow.
            // wait for this iteration to complete.
#if READY
            cl_command_queue clqueue;
            clErr = clGetEventInfo(mainKernelEvt,
                        CL_EVENT_COMMAND_QUEUE,
                        sizeof(clqueue),
                        &clqueue,
                        NULL);
            CL_CHECK_ERROR(clErr);
            std::cerr << "queue=" << queue << ", clqueue=" << clqueue << std::endl;

            cl_context clctxt;
            clErr = clGetEventInfo(mainKernelEvt,
                        CL_EVENT_CONTEXT,
                        sizeof(clctxt),
                        &clctxt,
                        NULL);
            CL_CHECK_ERROR(clErr);
            std::cerr << "ctx=" << context << ", clctxt=" << clctxt << std::endl;
            cl_command_type clct;
            clErr = clGetEventInfo(mainKernelEvt,
                        CL_EVENT_COMMAND_TYPE,
                        sizeof(clct),
                        &clct,
                        NULL );
            CL_CHECK_ERROR(clErr);
            std::cerr << "command type=" << clct << std::endl;
            cl_int cles;
            clErr = clGetEventInfo(mainKernelEvt,
                        CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(cles),
                        &cles,
                        NULL);
            CL_CHECK_ERROR(clErr);
            std::cerr << "execution status=" << cles << std::endl;

#endif // READY

            clErr = clWaitForEvents( 1, &mainKernelEvt );
            CL_CHECK_ERROR(clErr);
            ClearWaitEvents( waitEvents );
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
    clErr = clEnqueueReadBuffer( queue,
                                    *currData,
                                    CL_TRUE,    // blocking
                                    0,          // offset
                                    mtx.GetDataSize(),  // number of bytes
                                    mtx.GetFlatData(),  // store location
                                    waitEvents.size(),  // number of events to wait on
                                    waitEvents.empty() ? NULL : &waitEvents.front(), // events to wait on
                                    NULL ); // completion event
    CL_CHECK_ERROR(clErr);
    ClearWaitEvents( waitEvents );

    clReleaseMemObject( dataBuf1 );
    clReleaseMemObject( dataBuf2 );
}


template<class T>
void
OpenCLStencil<T>::DoPreIterationWork( cl_mem buf,
                                    cl_mem altBuf,
                                    Matrix2D<T>& mtx,
                                    unsigned int iter,
                                    cl_command_queue queue )
{
    // in single process version, nothing for us to do
}

