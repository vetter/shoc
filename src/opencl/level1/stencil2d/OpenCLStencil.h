#ifndef OPENCLSTENCIL_H
#define OPENCLSTENCIL_H

#include <vector>
#include "Stencil.h"
#include "shoc_compat_cas.h"
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

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

    cl::Context& context;
    cl::Device& device;
    cl::CommandQueue& queue;
    cl::Kernel kernel;

protected:
    cl::Kernel copyRectKernel;

    virtual void DoPreIterationWork( cl::Buffer& buf,
                                        cl::Buffer& altBuf,
                                        Matrix2D<T>& mtx,
                                        unsigned int iter,
                                        cl::CommandQueue& queue );

public:
    OpenCLStencil( T wCenter,
                    T wCardinal,
                    T wDiagonal,
                    size_t _lRows,
                    size_t _lCols,
                    cl::Device& dev,
                    cl::Context& ctx,
                    cl::CommandQueue& queue );

    virtual void operator()( Matrix2D<T>&, unsigned int nIters );
};

#endif // OPENCLSTENCIL_H
