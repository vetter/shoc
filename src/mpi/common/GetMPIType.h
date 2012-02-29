
#ifndef GET_MPI_TYPE_H
#define GET_MPI_TYPE_H

#include <mpi.h>

inline MPI_Datatype GetMPIType(const float&)  { return MPI_FLOAT; }
inline MPI_Datatype GetMPIType(const double&) { return MPI_DOUBLE; }
inline MPI_Datatype GetMPIType(const int&)    { return MPI_INT; }
inline MPI_Datatype GetMPIType(const long&)   { return MPI_LONG; }
inline MPI_Datatype GetMPIType(const char&)   { return MPI_CHAR; }

#endif
