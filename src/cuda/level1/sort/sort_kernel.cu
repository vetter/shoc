// This kernel code based on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.

#include <cuda.h>
#include "sort_kernel.h"

__device__ uint scanLSB(const uint val, uint* s_data)
{
    // Shared mem is 256 uints long, set first half to 0's
    int idx = threadIdx.x;
    s_data[idx] = 0;
    __syncthreads();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += blockDim.x; // += 128 in this case

    // Unrolled scan in local memory

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    s_data[idx] = val;     __syncthreads();
    t = s_data[idx -  1];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  2];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  4];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  8];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 16];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 32];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 64];  __syncthreads();
    s_data[idx] += t;      __syncthreads();

    return s_data[idx] - val;  // convert inclusive -> exclusive
}

__device__ uint4 scan4(uint4 idata, uint* ptr)
{
    uint4 val4 = idata;
    uint4 sum;

    // Scan the 4 elements in idata within this thread
    sum.x = val4.x;
    sum.y = val4.y + sum.x;
    sum.z = val4.z + sum.y;
    uint val = val4.w + sum.z;

    // Now scan those sums across the local work group
    val = scanLSB(val, ptr);

    val4.x = val;
    val4.y = val + sum.x;
    val4.z = val + sum.y;
    val4.w = val + sum.z;

    return val4;
}

//----------------------------------------------------------------------------
//
// radixSortBlocks sorts all blocks of data independently in shared
// memory.  Each thread block (CTA) sorts one block of 4*CTA_SIZE elements
//
// The radix sort is done in two stages.  This stage calls radixSortBlock
// on each block independently, sorting on the basis of bits
// (startbit) -> (startbit + nbits)
//----------------------------------------------------------------------------

__global__ void radixSortBlocks(const uint nbits, const uint startbit,
                              uint4* keysOut, uint4* valuesOut,
                              uint4* keysIn,  uint4* valuesIn)
{
    __shared__ uint sMem[512];

    // Get Indexing information
    const uint i = threadIdx.x + (blockIdx.x * blockDim.x);
    const uint tid = threadIdx.x;
    const uint localSize = blockDim.x;

    // Load keys and vals from global memory
    uint4 key, value;
    key = keysIn[i];
    value = valuesIn[i];

    // For each of the 4 bits
    for(uint shift = startbit; shift < (startbit + nbits); ++shift)
    {
        // Check if the LSB is 0
        uint4 lsb;
        lsb.x = !((key.x >> shift) & 0x1);
        lsb.y = !((key.y >> shift) & 0x1);
        lsb.z = !((key.z >> shift) & 0x1);
        lsb.w = !((key.w >> shift) & 0x1);

        // Do an exclusive scan of how many elems have 0's in the LSB
        // When this is finished, address.n will contain the number of
        // elems with 0 in the LSB which precede elem n
        uint4 address = scan4(lsb, sMem);

        __shared__ uint numtrue;

        // Store the total number of elems with an LSB of 0
        // to shared mem
        if (tid == localSize - 1)
        {
            numtrue = address.w + lsb.w;
        }
        __syncthreads();

        // Determine rank -- position in the block
        // If you are a 0 --> your position is the scan of 0's
        // If you are a 1 --> your position is calculated as below
        uint4 rank;
        const int idx = tid*4;
        rank.x = lsb.x ? address.x : numtrue + idx     - address.x;
        rank.y = lsb.y ? address.y : numtrue + idx + 1 - address.y;
        rank.z = lsb.z ? address.z : numtrue + idx + 2 - address.z;
        rank.w = lsb.w ? address.w : numtrue + idx + 3 - address.w;

        // Scatter keys into local mem
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = key.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = key.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = key.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = key.w;
        __syncthreads();

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
        key.x = sMem[tid];
        key.y = sMem[tid +     localSize];
        key.z = sMem[tid + 2 * localSize];
        key.w = sMem[tid + 3 * localSize];
        __syncthreads();

        // Scatter values into local mem
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = value.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = value.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = value.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = value.w;
        __syncthreads();

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
        value.x = sMem[tid];
        value.y = sMem[tid +     localSize];
        value.z = sMem[tid + 2 * localSize];
        value.w = sMem[tid + 3 * localSize];
        __syncthreads();
    }
    keysOut[i]   = key;
    valuesOut[i] = value;
}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each
// block counts the number of keys that fall into each radix in the group, and
// finds the starting offset of each radix in the block.  It then writes the
// radix counts to the counters array, and the starting offsets to the
// blockOffsets array.
//
//----------------------------------------------------------------------------
__global__ void findRadixOffsets(uint2* keys, uint* counters,
        uint* blockOffsets, uint startbit, uint numElements, uint totalBlocks)
{
    __shared__ uint  sStartPointers[16];
    extern __shared__ uint sRadix1[];

    uint groupId = blockIdx.x;
    uint localId = threadIdx.x;
    uint groupSize = blockDim.x;

    uint2 radix2;
    radix2 = keys[threadIdx.x + (blockIdx.x * blockDim.x)];

    sRadix1[2 * localId]     = (radix2.x >> startbit) & 0xF;
    sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;

    // Finds the position where the sRadix1 entries differ and stores start
    // index for each radix.
    if(localId < 16)
    {
        sStartPointers[localId] = 0;
    }
    __syncthreads();

    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId]] = localId;
    }
    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1])
    {
        sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
    }
    __syncthreads();

    if(localId < 16)
    {
        blockOffsets[groupId*16 + localId] = sStartPointers[localId];
    }
    __syncthreads();

    // Compute the sizes of each block.
    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId - 1]] =
            localId - sStartPointers[sRadix1[localId - 1]];
    }
    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] )
    {
        sStartPointers[sRadix1[localId + groupSize - 1]] =
            localId + groupSize - sStartPointers[sRadix1[localId +
                                                         groupSize - 1]];
    }

    if(localId == groupSize - 1)
    {
        sStartPointers[sRadix1[2 * groupSize - 1]] =
            2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
    }
    __syncthreads();

    if(localId < 16)
    {
        counters[localId * totalBlocks + groupId] = sStartPointers[localId];
    }
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets
// have been found. On compute version 1.1 and earlier GPUs, this code depends
// on SORT_BLOCK_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).
//----------------------------------------------------------------------------
__global__ void reorderData(uint  startbit,
                            uint  *outKeys,
                            uint  *outValues,
                            uint2 *keys,
                            uint2 *values,
                            uint  *blockOffsets,
                            uint  *offsets,
                            uint  *sizes,
                            uint  totalBlocks)
{
    uint GROUP_SIZE = blockDim.x;
    __shared__ uint2 sKeys2[256];
    __shared__ uint2 sValues2[256];
    __shared__ uint  sOffsets[16];
    __shared__ uint  sBlockOffsets[16];
    uint* sKeys1   = (uint*) sKeys2;
    uint* sValues1 = (uint*) sValues2;

    uint blockId = blockIdx.x;

    uint i = blockId * blockDim.x + threadIdx.x;

    sKeys2[threadIdx.x]   = keys[i];
    sValues2[threadIdx.x] = values[i];

    if(threadIdx.x < 16)
    {
        sOffsets[threadIdx.x]      = offsets[threadIdx.x * totalBlocks +
                                             blockId];
        sBlockOffsets[threadIdx.x] = blockOffsets[blockId * 16 + threadIdx.x];
    }
    __syncthreads();

    uint radix = (sKeys1[threadIdx.x] >> startbit) & 0xF;
    uint globalOffset = sOffsets[radix] + threadIdx.x - sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[threadIdx.x];
    outValues[globalOffset] = sValues1[threadIdx.x];

    radix = (sKeys1[threadIdx.x + GROUP_SIZE] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + threadIdx.x + GROUP_SIZE -
                   sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[threadIdx.x + GROUP_SIZE];
    outValues[globalOffset] = sValues1[threadIdx.x + GROUP_SIZE];

}

__device__ uint scanLocalMem(const uint val, uint* s_data)
{
    // Shared mem is 512 uints long, set first half to 0
    int idx = threadIdx.x;
    s_data[idx] = 0.0f;
    __syncthreads();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += blockDim.x; // += 256

    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    uint t;
    s_data[idx] = val;     __syncthreads();
    t = s_data[idx -  1];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  2];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  4];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  8];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 16];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 32];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 64];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 128]; __syncthreads();
    s_data[idx] += t;      __syncthreads();

    return s_data[idx-1];
}

__global__ void
scan(uint *g_odata, uint* g_idata, uint* g_blockSums, const int n,
     const bool fullBlock, const bool storeSum)
{
    __shared__ uint s_data[512];

    // Load data into shared mem
    uint4 tempData;
    uint4 threadScanT;
    uint res;
    uint4* inData  = (uint4*) g_idata;

    const int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int tid = threadIdx.x;
    const int i = gid * 4;

    // If possible, read from global mem in a uint4 chunk
    if (fullBlock || i + 3 < n)
    {
        // scan the 4 elems read in from global
        tempData       = inData[gid];
        threadScanT.x = tempData.x;
        threadScanT.y = tempData.y + threadScanT.x;
        threadScanT.z = tempData.z + threadScanT.y;
        threadScanT.w = tempData.w + threadScanT.z;
        res = threadScanT.w;
    }
    else
    {   // if not, read individual uints, scan & store in lmem
        threadScanT.x = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.y = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.x;
        threadScanT.z = ((i+2 < n) ? g_idata[i+2] : 0.0f) + threadScanT.y;
        threadScanT.w = ((i+3 < n) ? g_idata[i+3] : 0.0f) + threadScanT.z;
        res = threadScanT.w;
    }

    res = scanLocalMem(res, s_data);
    __syncthreads();

    // If we have to store the sum for the block, have the last work item
    // in the block write it out
    if (storeSum && tid == blockDim.x-1) {
        g_blockSums[blockIdx.x] = res + threadScanT.w;
    }

    // write results to global memory
    uint4* outData = (uint4*) g_odata;

    tempData.x = res;
    tempData.y = res + threadScanT.x;
    tempData.z = res + threadScanT.y;
    tempData.w = res + threadScanT.z;

    if (fullBlock || i + 3 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if ( i    < n) { g_odata[i]   = tempData.x;
        if ((i+1) < n) { g_odata[i+1] = tempData.y;
        if ((i+2) < n) { g_odata[i+2] = tempData.z; } } }
    }
}

__global__ void
vectorAddUniform4(uint *d_vector, const uint *d_uniforms, const int n)
{
    __shared__ uint uni[1];

    if (threadIdx.x == 0)
    {
        uni[0] = d_uniforms[blockIdx.x];
    }

    unsigned int address = threadIdx.x + (blockIdx.x *
            blockDim.x * 4);

    __syncthreads();

    // 4 elems per thread
    for (int i = 0; i < 4 && address < n; i++)
    {
        d_vector[address] += uni[0];
        address += blockDim.x;
    }
}
