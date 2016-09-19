#ifndef SUPPORT_H
#define SUPPORT_H

#include "hip_runtime.h"
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
    hipGetDevice(&device);
    CHECK_CUDA_ERROR();
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, device);
    CHECK_CUDA_ERROR();
    unsigned long total_bytes = deviceProp.totalGlobalMem;
    unsigned long avail_bytes = total_bytes;
    void* work;

    while (1) {
        hipMalloc(&work, avail_bytes);
        if (hipGetLastError() == hipSuccess) {
            break;
        }
        avail_bytes -= (1024*1024);
    }
    hipFree(work);
    CHECK_CUDA_ERROR();

    return avail_bytes;
}



#endif
