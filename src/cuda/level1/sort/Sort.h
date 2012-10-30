#ifndef SORT_H_
#define SORT_H_

typedef unsigned int uint;

static const int SORT_BLOCK_SIZE = 128;
static const int SCAN_BLOCK_SIZE = 256;
static const int SORT_BITS = 32;

void
radixSortStep(uint nbits, uint startbit, uint4* keys, uint4* values,
        uint4* tempKeys, uint4* tempValues, uint* counters,
        uint* countersSum, uint* blockOffsets, uint** scanBlockSums,
        uint numElements);

void
scanArrayRecursive(uint* outArray, uint* inArray, int numElements, int level,
        uint** blockSums);

bool
verifySort(uint *keys, uint* vals, const size_t size);

#ifdef __DEVICE_EMULATION__
#define __SYNC __syncthreads();
#else
#define __SYNC ;
#endif

#endif // SORT_H_
