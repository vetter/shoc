#ifndef OPENCLSTENCIL_H
#define OPENCLSTENCIL_H

#include <vector>
#include "Stencil.h"
#include "support.h"


// ****************************************************************************
// Class:  OpenCLStencil
//
// Purpose:
//   OpenCL implementation of 9-point stencil.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
class OpenCLStencil : public Stencil<T>
{
private:
    size_t lRows;
    size_t lCols;

    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_kernel kernel;

protected:
    cl_kernel copyRectKernel;

    virtual void DoPreIterationWork( cl_mem buf,
                                        cl_mem altBuf,
                                        Matrix2D<T>& mtx,
                                        unsigned int iter,
                                        cl_command_queue queue );

    void SetCopyRectKernelArgs( cl_mem dest,
                                int destOffset,
                                int destPitch,
                                cl_mem src,
                                int srcOffset,
                                int srcPitch,
                                int width,
                                int height );
 
    void SetStencilKernelArgs( cl_mem currData,
                                cl_mem newData,
                                int alignment,
                                T wCenter,
                                T wCardinal,
                                T wDiagonal,
                                size_t localDataSize );

    static void ClearWaitEvents( std::vector<cl_event>& waitEvents );

    cl_context  GetContext( void )      { return context; }

public:
    OpenCLStencil( T wCenter,
                    T wCardinal,
                    T wDiagonal,
                    size_t _lRows,
                    size_t _lCols,
                    cl_device_id dev,
                    cl_context ctx,
                    cl_command_queue queue );

    virtual void operator()( Matrix2D<T>&, unsigned int nIters );
};

#endif // OPENCLSTENCIL_H
