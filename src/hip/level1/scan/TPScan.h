#ifndef __TPSCAN_H
#define __TPSCAN_H

// When using MPICH and MPICH-derived MPI implementations, there is a
// naming conflict between stdio.h and MPI's C++ binding.
// Since we do not use the C++ MPI binding, we can avoid the ordering
// issue by ignoring the C++ MPI binding headers.
// This #define should be quietly ignored when using other MPI implementations.
#define MPICH_SKIP_MPICXX
#include "mpi.h"

// Templated wrapper for MPI_Exscan
template <class T>
inline void globalExscan(T* local_result, T* global_result);

template <>
inline void globalExscan(float* local_result, float* global_result)
{
   MPI_Exscan(local_result, global_result, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

template <>
inline void globalExscan(double* local_result, double* global_result)
{
   MPI_Exscan(local_result, global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

template<class T>
void
LaunchReduceKernel( int num_blocks,
                    int num_threads,
                    int smem_size,
                    T* d_idata,
                    T* d_odata,
                    int size );

template<class T>
void
LaunchTopScanKernel( int num_blocks,
                     int num_threads,
                     int smem_size,
                     T* d_block_sums,
                     int size );

template<class T, class vecT, int blockSize>
void
LaunchBottomScanKernel( int num_blocks,
                        int num_threads,
                        int smem_size,
                        T* g_idata,
                        T* g_odata,
                        T* d_block_sums,
                        int size );

#endif // __TPSCAN_H
