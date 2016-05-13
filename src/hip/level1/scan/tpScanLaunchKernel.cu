#include "scan_kernel.h"

template<class T>
void
LaunchReduceKernel( int num_blocks,
                    int num_threads,
                    int smem_size,
                    T* d_idata,
                    T* d_odata,
                    int size )
{
    // In CUDA 4.0 we will be able to remove this level of indirection
    // if we use the cuConfigureCall and cuLaunchKernel functions.
    reduce<T,256><<<num_blocks,num_threads,smem_size>>>
        (d_idata, d_odata, size);
}

template<class T>
void
LaunchTopScanKernel( int num_blocks,
                     int num_threads,
                     int smem_size,
                     T* d_block_sums,
                     int size )
{
    // In CUDA 4.0 we will be able to remove this level of indirection
    // if we use the cuConfigureCall and cuLaunchKernel functions.
    scan_single_block<T,256><<<num_blocks,num_threads,smem_size>>>
        (d_block_sums, size);
}

template<class T, class vecT, int blockSize>
void
LaunchBottomScanKernel( int num_blocks,
                        int num_threads,
                        int smem_size,
                        T* g_idata,
                        T* g_odata,
                        T* d_block_sums,
                        int size )
{
    // In CUDA 4.0 we will be able to remove this level of indirection
    // if we use the cuConfigureCall and cuLaunchKernel functions.
    bottom_scan<T, vecT, blockSize><<<num_blocks,num_threads,smem_size>>>(g_idata, g_odata,
        d_block_sums, size);
}

// Ensure that the template functions are instantiated
// Unlike the Stencil2D CUDA version that needs to instantiate objects,
// we need to instantiate template functions.  Declaration of the needed
// specializations seem to work for several recent versions of g++ that
// people are likely to be using underneath nvcc.
template void LaunchReduceKernel<float>( int, int, int, float*, float*, int );
template void LaunchReduceKernel<double>( int, int, int, double*, double*, int );

template void LaunchTopScanKernel<float>( int, int, int, float*, int );
template void LaunchTopScanKernel<double>( int, int, int, double*, int );

template void LaunchBottomScanKernel<float,float4,256>( int, int, int, float*, float*, float*, int );
template void LaunchBottomScanKernel<double,double4,256>( int, int, int, double*, double*, double*, int );

