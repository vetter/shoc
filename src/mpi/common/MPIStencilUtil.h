#ifndef STENCIL_UTILS_MPI_H
#define STENCIL_UTILS_MPI_H

#include "StencilUtil.h"


// ****************************************************************************
// Class:  MPIStencilValidater
//
// Purpose:
//   MPI version of stencil validator.
//
// Programmer:  Phil Roth
// Creation:    October 29, 2009
//
// ****************************************************************************
template<class T>
class MPIStencilValidater : public StencilValidater<T>
{
public:
    virtual void ValidateResult( const Matrix2D<T>& exp,
                const Matrix2D<T>& data,
                double valErrThreshold,
                unsigned int nValErrsToPrint ) const;
};


// ****************************************************************************
// Class:  MPIStencilTimingReporter
//
// Purpose:
//   MPI version of stencil timing reporter.
//
// Programmer:  Phil Roth
// Creation:    October 29, 2009
//
// ****************************************************************************
class MPIStencilTimingReporter : public StencilTimingReporter
{
public:
    virtual void ReportTimings( ResultDatabase& resultDB ) const;
};


#endif // STENCIL_UTILS_MPI_H
