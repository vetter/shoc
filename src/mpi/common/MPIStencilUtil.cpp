#include <mpi.h>
#include <sstream>
#include "MPIStencilUtil.h"
#include "ParallelResultDatabase.h"

template<class T>
void
MPIStencilValidater<T>::ValidateResult( const Matrix2D<T>& exp,
                const Matrix2D<T>& data,
                double valErrThreshold,
                unsigned int nValErrsToPrint ) const
{
    Validate<T> val( valErrThreshold );
    std::vector<ValidationErrorInfo<T> > validationErrors = val( exp, data );
    std::ostringstream valResultStr;

    // gather validation results to rank 0, who handles results
    int nValErrors = validationErrors.size();
    int totalValErrors = 0;
    MPI_Reduce( &nValErrors,        // input from each
                    &totalValErrors,    // output (only valid at root)
                    1,          // count
                    MPI_INT,   // datatype
                    MPI_SUM,   // reduction operation
                    0,          // root
                    MPI_COMM_WORLD );   // comm

    int cwrank;
    MPI_Comm_rank( MPI_COMM_WORLD, &cwrank );
    if( cwrank == 0 )
    {
        valResultStr << totalValErrors << " validation errors";

        if( (totalValErrors > 0) && (nValErrsToPrint > 0) )
        {
            unsigned int valErrPrintsRemaining = nValErrsToPrint;
            this->PrintValidationErrors( valResultStr, validationErrors, valErrPrintsRemaining );
            if( validationErrors.size() <= valErrPrintsRemaining )
            {
                // TODO do we want to collect validation errors from
                // other processes?
                valResultStr << " more validation errors in processes other than rank 0\n";
            }
        }
        std::cout << valResultStr.str() << std::endl;
    }
}




//  Modifications:
//    Jeremy Meredith, Wed Nov 10 14:20:47 EST 2010
//    Split timing reports into detailed and summary.  For
//    parallel code, don't report per-process values.
//
void
MPIStencilTimingReporter::ReportTimings( ResultDatabase& resultDB ) const
{
    ParallelResultDatabase pdb;
    pdb.MergeSerialDatabases( resultDB, MPI_COMM_WORLD );

    int cwrank;
    MPI_Comm_rank( MPI_COMM_WORLD, &cwrank );
    if( cwrank == 0 )
    {
        pdb.DumpSummary( std::cout );
    }
}

