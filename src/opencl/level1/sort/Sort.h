#ifndef _SORT_H
#define _SORT_H

static const int SORT_BLOCK_SIZE = 128;
static const int SORT_BITS = 32;

void radixSortStep(uint nbits, uint startbit, cl_mem counters,
        cl_mem countersSum, cl_mem blockOffsets, cl_mem* scanBlockSums,
        uint numElements, cl_kernel sortBlocks, cl_kernel findOffsets,
        cl_kernel reorder, cl_kernel scan, cl_kernel uniformAdd,
        cl_command_queue queue, cl_device_id dev);

#endif // _SORT_H
