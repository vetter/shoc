#ifndef MPICUDASTENCIL_H
#define MPICUDASTENCIL_H

#include <fstream>
#include <vector>
#include "CUDAStencil.h"
#include "MPI2DGridProgram.h"


// ****************************************************************************
// Class:  MPICUDAStencil
//
// Purpose:
//   MPI implementation of CUDA stencil
//
// Programmer:  Phil Roth
// Creation:    November 5, 2009
//
// ****************************************************************************
template<class T>
class MPICUDAStencil : public CUDAStencil<T>, public MPI2DGridProgram<T>
{
private:
    std::ofstream ofs;
    bool dumpData;

    virtual void DoPreIterationWork( T* currBuf,    // in device global memory
                                        T* altBuf,  // in device global memory
                                        Matrix2D<T>& mtx,
                                        unsigned int iter );

public:
    MPICUDAStencil( T _wCenter,
                    T _wCardinal,
                    T _wDiagonal,
                    size_t _lRows,
                    size_t _lCols,
                    size_t _mpiGridRows,
                    size_t _mpiGridCols,
                    unsigned int _nItersPerHaloExchange,
                    int _deviceIdx = 0,
                    bool dumpData = false );

    virtual void operator()( Matrix2D<T>&, unsigned int nIters );
};

#endif // MPICUDASTENCIL_H
