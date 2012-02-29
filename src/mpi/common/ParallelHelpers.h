#ifndef PARALLEL_HELPERS_H
#define PARALLEL_HELPERS_H

#include <mpi.h>
#include "GetMPIType.h"

// ****************************************************************************
// File:  ParallelHelpers.h
//
// Purpose:
//   Various C++ encapsulations of MPI routines
//
// Programmer:  Jeremy Meredith
// Creation:    August 14, 2009
//
// Modifications:
//   Jeremy Meredith, Tue Jan 12 14:39:40 EST 2010
//   Added ParAllGather.
//
// ****************************************************************************

template<class T>
T ParSumAcrossProcessors(const T &val, MPI_Comm comm)
{
    T newval;
    MPI_Allreduce((void*)&val, &newval, 1,
                  GetMPIType(val), MPI_SUM, comm);
    return newval;
}

template<class T>
vector<T> ParGather(const T &val, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    vector<T> retval;
    if (rank==0)
        retval.resize(size);
    MPI_Datatype t = GetMPIType(val);
    MPI_Gather((void*)(&val), 1, t,
               &(retval[0]), 1, t,
               0, comm);
    return retval;
}

template<class T>
vector<T> ParAllGather(const T &val, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_size(comm, &size);
    vector<T> retval;
    retval.resize(size);
    MPI_Datatype t = GetMPIType(val);
    MPI_Allgather((void*)(&val), 1, t,
                  &(retval[0]), 1, t,
                  comm);
    return retval;
}

#endif
