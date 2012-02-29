#include "CUDAStencil.h"

template<class T>
CUDAStencil<T>::CUDAStencil( T _wCenter,
                    T _wCardinal,
                    T _wDiagonal,
                    int _device )
  : Stencil<T>( _wCenter, _wCardinal, _wDiagonal ),
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

