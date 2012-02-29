#ifndef STENCIL_UTILS_SERIAL_H
#define STENCIL_UTILS_SERIAL_H

#include "StencilUtil.h"


// ****************************************************************************
// Class:  SerialStencilValidater
//
// Purpose:
//   Single-processor version of stencil validator.
//
// Programmer:  Phil Roth
// Creation:    October 29, 2009
//
// ****************************************************************************
template<class T>
class SerialStencilValidater : public StencilValidater<T>
{
public:
    virtual void ValidateResult( const Matrix2D<T>& exp,
                const Matrix2D<T>& data,
                double valErrThreshold,
                unsigned int nValErrsToPrint ) const;
};


// ****************************************************************************
// Class:  SerialStencilTimingReporter
//
// Purpose:
//   Single-processor version of stencil timing reporter.
//
// Programmer:  Phil Roth
// Creation:    October 29, 2009
//
// ****************************************************************************
class SerialStencilTimingReporter : public StencilTimingReporter
{
public:
    virtual void ReportTimings( ResultDatabase& resultDB ) const;
};


#endif // STENCIL_UTILS_SERIAL_H
