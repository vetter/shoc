#include <assert.h>
#include <math.h>
#include "ValidateMatrix2D.h"


template<class T>
std::vector<ValidationErrorInfo<T> >
Validate<T>::operator()( const Matrix2D<T>& s, const Matrix2D<T>& t )
{
    std::vector<ValidationErrorInfo<T> > ret;

    // ensure matrices are same shape
    assert( (s.GetNumRows() == t.GetNumRows()) && (s.GetNumColumns() == t.GetNumColumns()) );

    for( unsigned int i = 0; i < s.GetNumRows(); i++ )
    {
        for( unsigned int j = 0; j < s.GetNumColumns(); j++ )
        {
            T expVal = s.GetConstData()[i][j];
            T actualVal = t.GetConstData()[i][j];
            T delta = fabsf( actualVal - expVal );
            T relError = (expVal != 0.0f) ? delta / expVal : 0.0f;

            if( relError > relErrThreshold )
            {
                ret.push_back( ValidationErrorInfo<T>( i, j, actualVal, expVal, relError ) );
            }
        }
    }

    return ret;
}


