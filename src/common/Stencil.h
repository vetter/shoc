#ifndef STENCIL_H
#define STENCIL_H

#include <string>
#include <functional>
#include "Matrix2D.h"

// ****************************************************************************
// Class:  Stencil
//
// Purpose:
//   9-point stencil.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
class Stencil : public std::binary_function<Matrix2D<T>&, unsigned int, void>
{
protected:
    T wCenter;
    T wCardinal;
    T wDiagonal;

protected:
    T GetCenterWeight( void ) const { return wCenter; }
    T GetCardinalWeight( void ) const { return wCardinal; }
    T GetDiagonalWeight( void ) const { return wDiagonal; }

public:
    Stencil( T _wCenter,
                T _wCardinal,
                T _wDiagonal )
      : wCenter( _wCenter ),
        wCardinal( _wCardinal ),
        wDiagonal( _wDiagonal )
    {
        // nothing else to do
    }

    virtual ~Stencil( void )
    {
        // nothing to do
    }


    /*
     * This is a 9-point stencil using three weights:
     *   wCenter is applied to the stencil 'center'
     *   wCardinal is applied to the sum of the stencil NSEW values
     *   wDiagonal is applied to the sum of the stencil diagonal values
     *
     * note two things:
     *   We use the overall boundary values but do not update them.
     *   We apply wCardinal and wDiagonal *only* to the sum of the NSEW and
     *     diagonal values. We don't do any other averaging, etc.
     */
    virtual void operator()( Matrix2D<T>& m, unsigned int nIters ) = 0;
};

#endif // STENCIL_H
