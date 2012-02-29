#ifndef HOSTSTENCIL_H
#define HOSTSTENCIL_H

#include "Stencil.h"

// ****************************************************************************
// Class:  HostStencil
//
// Purpose:
//   Stencils for hosts.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
class HostStencil : public Stencil<T>
{
protected:
    virtual void DoPreIterationWork( Matrix2D<T>& mtx, unsigned int iter );

public:
    HostStencil( T wCenter,
                        T wCardinal,
                        T wDiagonal )
      : Stencil<T>( wCenter, wCardinal, wDiagonal )
    {
        // nothing else to do
    }

    virtual void operator()( Matrix2D<T>&, unsigned int nIters );
};

#endif /* HOSTSTENCIL_H */
