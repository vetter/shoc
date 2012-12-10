#ifndef OPENACCSTENCIL_H
#define OPENACCSTENCIL_H

#include <vector>
#include "Stencil.h"


// ****************************************************************************
// Class:  OpenACCStencil
//
// Purpose:
//   OpenACC implementation of 9-point stencil.
//
// Programmer:  Phil Roth
// Creation:    2012-12-06
//
// ****************************************************************************
template<class T>
class OpenACCStencil : public Stencil<T>
{
private:
    // As of now (December 2012), we only support the PGI compiler
    // for OpenACC programs.  The PGI compiler does not support C++ programs,
    // so we can't implement the OpenACC stencil operator as part of 
    // this templatized class.  Instead, we call out to one of two
    // C functions depending on our data type.  Because we call through
    // a function pointer we have to ensure the two functions have the
    // same signature, hence our arrays are passed through void pointer
    // arguments.
    void (*applyfunc)( void* vdata, 
                        unsigned int nRows, 
                        unsigned int nCols, 
                        unsigned int nPaddedCols, 
                        unsigned int nIters, 
                        unsigned int nItersPerExchange,
                        void* vwCenter,
                        void* vwCardinal, 
                        void* vwDiagonal,
                        void (*preIterBlockCB)( void* cbData ),
                        void* cbData );

protected:
    virtual void ApplyOperator( Matrix2D<T>&, 
                            unsigned int nIters, 
                            unsigned int nItersPerBlock = 0,    // 0 => nIters
                            void (*preIterBlockCB)( void* cbData ) = NULL,
                            void* cbData = NULL );


public:
    OpenACCStencil( T wCenter,
                    T wCardinal,
                    T wDiagonal );

    virtual void operator()( Matrix2D<T>&, unsigned int nIters );
};

#endif // OPENACCSTENCIL_H
