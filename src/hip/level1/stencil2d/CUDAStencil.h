#ifndef CUDASTENCIL_H
#define CUDASTENCIL_H

#include "Stencil.h"

// ****************************************************************************
// Class:  CUDAStencil
//
// Purpose:
//   CUDA implementation of 9-point stencil.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
class CUDAStencil : public Stencil<T>
{
private:
    size_t lRows;
    size_t lCols;
    int device;

protected:
    virtual void DoPreIterationWork( T* currBuf,    // in device global memory
                                        T* altBuf,  // in device global memory
                                        Matrix2D<T>& mtx,
                                        unsigned int iter );

public:
    CUDAStencil( T _wCenter,
                    T _wCardinal,
                    T _wDiagonal,
                    size_t _lRows,
                    size_t _lCols,
                    int _device );

    virtual void operator()( Matrix2D<T>&, unsigned int nIters );
};

#endif /* CUDASTENCIL_H */
