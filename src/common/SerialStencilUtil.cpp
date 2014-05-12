#include <iostream>
#include <sstream>
#include "SerialStencilUtil.h"
#include "ValidateMatrix2D.h"


template<class T>
void
SerialStencilValidater<T>::ValidateResult( const Matrix2D<T>& exp,
                const Matrix2D<T>& data,
                double valErrThreshold,
                unsigned int nValErrsToPrint ) const
{
    Validate<T> val( valErrThreshold );
    std::vector<ValidationErrorInfo<T> > validationErrors = val( exp, data );
    std::ostringstream valResultStr;

    valResultStr << validationErrors.size() << " validation errors";
    if( (validationErrors.size() > 0) && (nValErrsToPrint > 0) )
    {
        this->PrintValidationErrors( valResultStr, validationErrors, nValErrsToPrint );
    }
    std::cout << valResultStr.str() << std::endl;
}



//  Modifications:
//    Jeremy Meredith, Wed Nov 10 14:20:47 EST 2010
//    Split timing reports into detailed and summary.  For
//    serial code, we report all trial values.
//
void
SerialStencilTimingReporter::ReportTimings( ResultDatabase& resultDB ) const
{
    resultDB.DumpDetailed( std::cout );
}

