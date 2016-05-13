#include "CUDAStencil.h"

template<class T>
CUDAStencil<T>::CUDAStencil( T _wCenter,
                    T _wCardinal,
                    T _wDiagonal,
                    size_t _lRows,
                    size_t _lCols,
                    int _device )
  : Stencil<T>( _wCenter, _wCardinal, _wDiagonal ),
    lRows( _lRows ),
    lCols( _lCols ),
    device( _device )
{
    // nothing else to do
}

template<class T>
void
CUDAStencil<T>::DoPreIterationWork( T* currBuf, // in device global memory
                                    T* altBuf,  // in device global memory
                                    Matrix2D<T>& mtx,
                                    unsigned int iter )
{
    // in single-process version, nothing for us to do
}

