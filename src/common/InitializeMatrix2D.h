#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <functional>
#include "Matrix2D.h"

// ****************************************************************************
// Class:  Initialize
//
// Purpose:
//   Initialize 2D matrices.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
class Initialize : public std::unary_function<Matrix2D<T>&, void>
{
private:
    long seed;
    unsigned int haloWidth; // width of halo
    T haloVal;          // value to use for halo
    int rowPeriod;          // period for row values
    int colPeriod;          // period for column values

public:
    Initialize( long int _seed,
                unsigned int _halo = 1,
                T _haloVal = 0,
                int _rowPeriod = -1,
                int _colPeriod = -1 )
      : seed( _seed ),
        haloWidth( _halo ),
        haloVal( _haloVal ),
        rowPeriod( _rowPeriod ),
        colPeriod( _colPeriod )
    {
        // nothing else to do
    }

    void operator()( Matrix2D<T>& mtx );
};

#endif // INITIALIZE_H
