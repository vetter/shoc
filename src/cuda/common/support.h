#ifndef SUPPORT_H
#define SUPPORT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudacommon.h"
#include <iostream>
using std::cin;
using std::cout;

// ****************************************************************************
// Method:  findAvailBytes
//
// Purpose: returns maximum number of bytes *allocatable* (likely less than
//          device memory size) on the device.
//
// Arguments: None.
//
// Programmer:  Collin McCurdy
// Creation:    June 8, 2010
//
// ****************************************************************************
inline unsigned long
findAvailBytes(void)
{
    int device;
    cudaGetDevice(&device);
    CHECK_CUDA_ERROR();
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    CHECK_CUDA_ERROR();
    unsigned long total_bytes = deviceProp.totalGlobalMem;
    unsigned long avail_bytes = total_bytes;
    void* work;

    while (1) {
        cudaMalloc(&work, avail_bytes);
        if (cudaGetLastError() == cudaSuccess) {
            break;
        }
        avail_bytes -= (1024*1024);
    }
    cudaFree(work);
    CHECK_CUDA_ERROR();

    return avail_bytes;
}



#endif
