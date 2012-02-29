#ifndef STENCIL_UTILS_H
#define STENCIL_UTILS_H

#include "Matrix2D.h"
#include "ResultDatabase.h"
#include "ValidateMatrix2D.h"


// ****************************************************************************
// Class:  StencilValidater
//
// Purpose:
//   Validate results of stencil operations and print errors.
//
// Programmer:  Phil Roth
// Creation:    October 29, 2009
//
// ****************************************************************************
template<class T>
class StencilValidater
{
protected:
    void PrintValidationErrors( std::ostream& s,
                const std::vector<ValidationErrorInfo<T> >& validationErrors,
                unsigned int nValErrsToPrint ) const;
public:
    virtual void ValidateResult( const Matrix2D<T>& exp,
                const Matrix2D<T>& data,
                double valErrThreshold,
                unsigned int nValErrsToPrint ) const = 0;
};


// ****************************************************************************
// Class:  StencilTimingReporter
//
// Purpose:
//   Report timing results of stencil operations.
//
// Programmer:  Phil Roth
// Creation:    October 29, 2009
//
// ****************************************************************************
class StencilTimingReporter
{
public:
    virtual void ReportTimings( ResultDatabase& resultDB ) const = 0;
};


#endif // STENCIL_UTILS_H
