#ifndef VALIDATE_H
#define VALIDATE_H

#include <functional>
#include <vector>
#include "Matrix2D.h"


// ****************************************************************************
// Struct:  ValidationErrorInfo
//
// Purpose:
//   Stores information about validation errors originating in a 2D grid.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
struct ValidationErrorInfo
{
    int i;
    int j;
    T val;
    T exp;
    double relErr;

    ValidationErrorInfo( int _i, int _j,
                            T _val,
                            T _exp,
                            double _relErr )
      : i( _i ),
        j( _j ),
        val( _val ),
        exp( _exp ),
        relErr( _relErr )
    {
        // nothing else to do
    }
};

// ****************************************************************************
// Class:  Validate
//
// Purpose:
//   Compares 2D matrices.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
class Validate : public std::binary_function<const Matrix2D<T>&, const Matrix2D<T>&, std::vector<ValidationErrorInfo<T> > >
{
private:
    double relErrThreshold;

public:
    Validate( double _relErrThreshold )
      : relErrThreshold( _relErrThreshold )
    {
        // nothing else to do
    }

    std::vector<ValidationErrorInfo<T> > operator()( const Matrix2D<T>& s, const Matrix2D<T>& t );
};

#endif // VALIDATE_H
