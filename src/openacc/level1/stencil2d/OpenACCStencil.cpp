#include <cassert>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include "Stencil.h"
#include "OpenACCStencil.h"
#include "InvalidArgValue.h"


// As of now (December 2012), we only support the PGI compiler
// for OpenACC programs.  The PGI compiler does not support C++ programs,
// so we can't implement the OpenACC stencil operator as part of 
// this templatized class.  Instead, we call out to one of two
// C functions depending on our data type.  Because we call through
// a function pointer we have to ensure the two functions have the
// same signature, hence our arrays are passed through void pointer
// arguments.
extern "C" void ApplyFloatStencil( void* data, 
                                    unsigned int nRows, 
                                    unsigned int nCols,
                                    unsigned int nPaddedCols, 
                                    unsigned int nIters, 
                                    unsigned int nItersPerExchange,
                                    void* vwCenter,
                                    void* vwCardinal,
                                    void* vwDiagonal,
                                    void (*preIterBlockFunc)(void* cbData),
                                    void* cbData );
extern "C" void ApplyDoubleStencil( void* data, 
                                    unsigned int nRows, 
                                    unsigned int nCols,
                                    unsigned int nPaddedCols, 
                                    unsigned int nIters, 
                                    unsigned int nItersPerExchange, 
                                    void* vwCenter,
                                    void* vwCardinal,
                                    void* vwDiagonal,
                                    void (*preIterBlockCB)(void* cbData),
                                    void* cbData );


template<class T>
OpenACCStencil<T>::OpenACCStencil( T wCenter,
                    T wCardinal,
                    T wDiagonal )
  : Stencil<T>( wCenter, wCardinal, wDiagonal )
{
    // Determine which of the Apply* functions we need to use.
    // NOTE: this test is not bullet proof - e.g., a uint64_t would 
    // be recognized as a 64-bit double.
    if( sizeof(T) == sizeof(double) )
    {
        applyfunc = ApplyDoubleStencil;
    }
    else if( sizeof(T) == sizeof(float))
    {
        applyfunc = ApplyFloatStencil;
    }
    else
    {
        std::cerr << "Stencil instantiated for type T that is not float or double" << std::endl;
    }
}





template<class T>
void
OpenACCStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    this->ApplyOperator( mtx, nIters );
}


template<class T>
void
OpenACCStencil<T>::ApplyOperator( Matrix2D<T>& mtx,
                                    unsigned int nIters,
                                    unsigned int nItersPerBlock,
                                    void (*preIterBlockCB)(void* cbData),
                                    void* cbData )
{
    T wCenter = this->GetCenterWeight();
    T wCardinal = this->GetCardinalWeight();
    T wDiagonal = this->GetDiagonalWeight();

    if( nItersPerBlock == 0 )
    {
        nItersPerBlock = nIters;
    }

    // See comment in constructor as to why we don't implement the
    // stencil operation here.
    T* data = mtx.GetFlatData();
    unsigned int nRows = mtx.GetNumRows();
    unsigned int nCols = mtx.GetNumColumns();
    unsigned int nPaddedCols = mtx.GetNumPaddedColumns();

    (*applyfunc)( data,
                    nRows,
                    nCols,
                    nPaddedCols,
                    nIters,
                    nItersPerBlock,
                    &wCenter,
                    &wCardinal,
                    &wDiagonal,
                    preIterBlockCB,
                    cbData );
}


