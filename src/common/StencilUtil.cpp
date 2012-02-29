#include "StencilUtil.h"


template<class T>
void
StencilValidater<T>::PrintValidationErrors( std::ostream& s,
                    const std::vector<ValidationErrorInfo<T> >& validationErrors,
                    unsigned int nValErrsToPrint ) const
{
    unsigned int nErrorsPrinted = 0;
    for( typename std::vector<ValidationErrorInfo<T> >::const_iterator iter = validationErrors.begin();
            iter != validationErrors.end();
            iter++ )
    {
        if( nErrorsPrinted <= nValErrsToPrint )
        {
            s << "out[" << iter->i
                << "][" << iter->j
                << "]=" << iter->val
                << ", expected " << iter->exp
                << ", relErr " << iter->relErr
                << '\n';
        }
        nErrorsPrinted++;
    }
}

