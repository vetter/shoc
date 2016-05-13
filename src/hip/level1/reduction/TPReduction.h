#ifndef TPREDUCTION_H_
#define TPREDUCTION_H_

#include "mpi.h"

template<class T>
void RunTestLaunchKernel(int num_blocks,
                         int num_threads,
                         int smem_size,
                         T* d_idata,
                         T* d_odata,
                         int size );

// Template specializations for MPI allreduce call.
template <class T>
inline void globalReduction(T* local_result, T* global_result);

template <>
inline void globalReduction(float* local_result, float* global_result)
{
   MPI_Allreduce(local_result, global_result, 1, MPI_FLOAT,
           MPI_SUM, MPI_COMM_WORLD);
}

template <>
inline void globalReduction(double* local_result, double* global_result)
{
   MPI_Allreduce(local_result, global_result, 1, MPI_DOUBLE,
           MPI_SUM, MPI_COMM_WORLD);
}

#endif // TPREDUCTION_H_
