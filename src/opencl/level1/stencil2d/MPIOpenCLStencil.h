#ifndef MPIOPENCLSTENCIL_H
#define MPIOPENCLSTENCIL_H

#include <fstream>
#include <vector>
#include "OpenCLStencil.h"
#include "MPI2DGridProgram.h"
#include "shoc_compat_cas.h"
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"


// ****************************************************************************
// Class:  MPIOpenCLStencil
//
// Purpose:
//   MPI implementation of OpenCL stencil
//
// Programmer:  Phil Roth
// Creation:    November 5, 2009
//
// ****************************************************************************
template<class T>
class MPIOpenCLStencil : public OpenCLStencil<T>, public MPI2DGridProgram<T>
{
private:
    std::ofstream ofs;
    bool dumpData;

    T* eData;
    T* wData;

    virtual void DoPreIterationWork( cl::Buffer& buf,
                                        cl::Buffer& altbuf,
                                        Matrix2D<T>& mtx,
                                        unsigned int iter,
                                        cl::CommandQueue& queue );

public:
    MPIOpenCLStencil( T wCenter,
                    T wCardinal,
                    T wDiagonal,
                    size_t _lRows,
                    size_t _lCols,
                    size_t _mpiGridRows,
                    size_t _mpiGridCols,
                    unsigned int _nItersPerHaloExchange,
                    cl::Device& dev,
                    cl::Context& ctx,
                    cl::CommandQueue& queue,
                    bool _dumpData = false );
    virtual ~MPIOpenCLStencil( void );

    virtual void operator()( Matrix2D<T>&, unsigned int nIters );
};

#endif // MPIOPENCLSTENCIL_H
