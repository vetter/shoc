#ifndef SORT_KERNEL_H_
#define SORT_KERNEL_H_

#include <cuda.h>

#define WARP_SIZE 32
#define SORT_BLOCK_SIZE 128
#define SCAN_BLOCK_SIZE 256

typedef unsigned int uint;

__global__ void radixSortBlocks(uint nbits, uint startbit, uint4* keysOut,
        uint4* valuesOut, uint4* keysIn, uint4* valuesIn);

__global__ void findRadixOffsets(uint2* keys, uint* counters,
        uint* blockOffsets, uint startbit, uint numElements, uint totalBlocks);

__global__ void reorderData(uint startbit, uint *outKeys, uint *outValues,
        uint2 *keys, uint2 *values, uint *blockOffsets, uint *offsets,
        uint *sizes, uint totalBlocks);

// Scan Kernels
__global__ void vectorAddUniform4(uint *d_vector, const uint *d_uniforms,
                                  const int n);

__global__ void scan(uint *g_odata, uint *g_idata, uint *g_blockSums,
        const int n, const bool fullBlock, const bool storeSum);

#endif // SORT_KERNEL_H_
