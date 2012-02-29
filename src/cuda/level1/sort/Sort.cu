#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cudacommon.h"
#include <cassert>
#include <iostream>
#include <vector>
#include "Sort.h"
#include "sort_kernel.h"
#include "OptionParser.h"
#include "ResultDatabase.h"

using namespace std;

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    ;
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the radix sort benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing, results are stored in resultDB
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{

    //Number of key-value pairs to sort, must be a multiple of 1024
    int probSizes[4] = { 1, 8, 48, 96 };

    int size = probSizes[op.getOptionInt("size")-1];
    // Convert to MB
    size = (size * 1024 * 1024) / sizeof(uint);

    // Size of the keys & vals buffers in bytes
    uint bytes = size * sizeof(uint);

    // create input data on CPU
    uint *hKeys;
    uint *hVals;
    cudaMallocHost((void**)&hKeys, bytes);
    cudaMallocHost((void**)&hVals, bytes);

    // Allocate space for block sums in the scan kernel.
    uint numLevelsAllocated = 0;
    uint maxNumScanElements = size;
    uint numScanElts = maxNumScanElements;
    uint level = 0;

    do
    {
        uint numBlocks = max(1, (int) ceil((float) numScanElts / (4
                * SCAN_BLOCK_SIZE)));
        if (numBlocks > 1)
        {
            level++;
        }
        numScanElts = numBlocks;
    }
    while (numScanElts > 1);

    uint** scanBlockSums = (uint**) malloc((level + 1) * sizeof(uint*));
    assert(scanBlockSums != NULL);
    numLevelsAllocated = level + 1;
    numScanElts = maxNumScanElements;
    level = 0;

    do
    {
        uint numBlocks = max(1, (int) ceil((float) numScanElts / (4
                * SCAN_BLOCK_SIZE)));
        if (numBlocks > 1)
        {
            // Malloc device mem for block sums
            CUDA_SAFE_CALL(cudaMalloc((void**)&(scanBlockSums[level]),
                    numBlocks*sizeof(uint)));
            level++;
        }
        numScanElts = numBlocks;
    }
    while (numScanElts > 1);

    CUDA_SAFE_CALL(cudaMalloc((void**)&(scanBlockSums[level]),
            sizeof(uint)));

    // Allocate device mem for sorting kernels
    uint* dKeys, *dVals, *dTempKeys, *dTempVals;

    CUDA_SAFE_CALL(cudaMalloc((void**)&dKeys, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dVals, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dTempKeys, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dTempVals, bytes));

    // Each thread in the sort kernel handles 4 elements
    size_t numSortGroups = size / (4 * SORT_BLOCK_SIZE);

    uint* dCounters, *dCounterSums, *dBlockOffsets;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dCounters, WARP_SIZE
            * numSortGroups * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dCounterSums, WARP_SIZE
            * numSortGroups * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dBlockOffsets, WARP_SIZE
            * numSortGroups * sizeof(uint)));

    int iterations = op.getOptionInt("passes");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int it = 0; it < iterations; it++)
    {
        // Initialize host memory to some pattern
        for (uint i = 0; i < size; i++)
        {
            hKeys[i] = hVals[i] = i % 1024;
        }

        // Copy inputs to GPU
        double transferTime = 0.;
        cudaEventRecord(start, 0);
        CUDA_SAFE_CALL(cudaMemcpy(dKeys, hKeys, bytes, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(dVals, hVals, bytes, cudaMemcpyHostToDevice));
        cudaEventRecord(stop, 0);
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        transferTime += elapsedTime * 1.e-3; // convert to seconds

        cudaEventRecord(start, 0);
        // Perform Radix Sort (4 bits at a time)
        for (int i = 0; i < SORT_BITS; i += 4)
        {
            radixSortStep(4, i, (uint4*)dKeys, (uint4*)dVals,
                    (uint4*)dTempKeys, (uint4*)dTempVals, dCounters,
                    dCounterSums, dBlockOffsets, scanBlockSums, size);
        }
        cudaEventRecord(stop, 0);
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&elapsedTime, start, stop);
        double kernelTime = elapsedTime * 1.e-3;

        // Readback data from device
        cudaEventRecord(start, 0);
        CUDA_SAFE_CALL(cudaMemcpy(hKeys, dKeys, bytes, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(hVals, dVals, bytes, cudaMemcpyDeviceToHost));
        cudaEventRecord(stop, 0);
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&elapsedTime, start, stop);
        transferTime += elapsedTime * 1.e-3;

        // Test to make sure data was sorted properly, if not, return
        if (! verifySort(hKeys, hVals, size))
        {
            return;
        }

        char atts[1024];
        sprintf(atts, "%ditems", size);
        double gb = (bytes * 2.) / (1000. * 1000. * 1000.);
        resultDB.AddResult("Sort-Rate", atts, "GB/s", gb / kernelTime);
        resultDB.AddResult("Sort-Rate_PCIe", atts, "GB/s",
                gb / (kernelTime + transferTime));
        resultDB.AddResult("Sort-Rate_Parity", atts, "N",
                transferTime / kernelTime);
    }
    // Clean up
    for (int i = 0; i < numLevelsAllocated; i++)
    {
        CUDA_SAFE_CALL(cudaFree(scanBlockSums[i]));
    }
    CUDA_SAFE_CALL(cudaFree(dKeys));
    CUDA_SAFE_CALL(cudaFree(dVals));
    CUDA_SAFE_CALL(cudaFree(dTempKeys));
    CUDA_SAFE_CALL(cudaFree(dTempVals));
    CUDA_SAFE_CALL(cudaFree(dCounters));
    CUDA_SAFE_CALL(cudaFree(dCounterSums));
    CUDA_SAFE_CALL(cudaFree(dBlockOffsets));
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));

    free(scanBlockSums);
    CUDA_SAFE_CALL(cudaFreeHost(hKeys));
    CUDA_SAFE_CALL(cudaFreeHost(hVals));
}

// ****************************************************************************
// Function: radixSortStep
//
// Purpose:
//   This function performs a radix sort, using bits startbit to
//   (startbit + nbits).  It is designed to sort by 4 bits at a time.
//   It also reorders the data in the values array based on the sort.
//
// Arguments:
//      nbits: the number of key bits to use
//      startbit: the bit to start on, 0 = lsb
//      keys: the input array of keys
//      values: the input array of values
//      tempKeys: temporary storage, same size as keys
//      tempValues: temporary storage, same size as values
//      counters: storage for the index counters, used in sort
//      countersSum: storage for the sum of the counters
//      blockOffsets: storage used in sort
//      scanBlockSums: input to Scan, see below
//      numElements: the number of elements to sort
//
// Returns: nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void radixSortStep(uint nbits, uint startbit, uint4* keys, uint4* values,
        uint4* tempKeys, uint4* tempValues, uint* counters,
        uint* countersSum, uint* blockOffsets, uint** scanBlockSums,
        uint numElements)
{
    // Threads handle either 4 or two elements each
    const size_t radixGlobalWorkSize   = numElements / 4;
    const size_t findGlobalWorkSize    = numElements / 2;
    const size_t reorderGlobalWorkSize = numElements / 2;

    // Radix kernel uses block size of 128, others use 256 (same as scan)
    const size_t radixBlocks   = radixGlobalWorkSize   / SORT_BLOCK_SIZE;
    const size_t findBlocks    = findGlobalWorkSize    / SCAN_BLOCK_SIZE;
    const size_t reorderBlocks = reorderGlobalWorkSize / SCAN_BLOCK_SIZE;

    radixSortBlocks
        <<<radixBlocks, SORT_BLOCK_SIZE, 4 * sizeof(uint)*SORT_BLOCK_SIZE>>>
        (nbits, startbit, tempKeys, tempValues, keys, values);

    findRadixOffsets
        <<<findBlocks, SCAN_BLOCK_SIZE, 2 * SCAN_BLOCK_SIZE*sizeof(uint)>>>
        ((uint2*)tempKeys, counters, blockOffsets, startbit, numElements,
         findBlocks);

    scanArrayRecursive(countersSum, counters, 16*reorderBlocks, 0,
            scanBlockSums);

    reorderData<<<reorderBlocks, SCAN_BLOCK_SIZE>>>
        (startbit, (uint*)keys, (uint*)values, (uint2*)tempKeys,
        (uint2*)tempValues, blockOffsets, countersSum, counters,
        reorderBlocks);
}

void scanArrayRecursive(uint* outArray, uint* inArray, int numElements,
        int level, uint** blockSums)
{
    // Kernels handle 8 elems per thread
    unsigned int numBlocks = max(1,
            (unsigned int)ceil((float)numElements/(4.f*SCAN_BLOCK_SIZE)));
    unsigned int sharedEltsPerBlock = SCAN_BLOCK_SIZE * 2;
    unsigned int sharedMemSize = sizeof(uint) * sharedEltsPerBlock;

    bool fullBlock = (numElements == numBlocks * 4 * SCAN_BLOCK_SIZE);

    dim3 grid(numBlocks, 1, 1);
    dim3 threads(SCAN_BLOCK_SIZE, 1, 1);

    // execute the scan
    if (numBlocks > 1)
    {
        scan<<<grid, threads, sharedMemSize>>>
           (outArray, inArray, blockSums[level], numElements, fullBlock, true);
    } else
    {
        scan<<<grid, threads, sharedMemSize>>>
           (outArray, inArray, blockSums[level], numElements, fullBlock, false);
    }
    if (numBlocks > 1)
    {
        scanArrayRecursive(blockSums[level], blockSums[level],
                numBlocks, level + 1, blockSums);
        vectorAddUniform4<<< grid, threads >>>
                (outArray, blockSums[level], numElements);
    }
}

// ****************************************************************************
// Function: verifySort
//
// Purpose:
//   Simple cpu routine to verify device results
//
// Arguments:
//
//
// Returns:  nothing, prints relevant info to stdout
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
bool verifySort(uint *keys, uint* vals, const size_t size)
{
    bool passed = true;

    for (unsigned int i = 0; i < size - 1; i++)
    {
        if (keys[i] > keys[i + 1])
        {
            passed = false;
#ifdef VERBOSE_OUTPUT
            cout << "Failure: at idx: " << i << endl;
            cout << "Key: " << keys[i] << " Val: " << vals[i] << endl;
            cout << "Idx: " << i + 1 << " Key: " << keys[i + 1] << " Val: "
                    << vals[i + 1] << endl;
#endif
        }
    }
    cout << "Test ";
    if (passed)
        cout << "Passed" << endl;
    else
        cout << "Failed" << endl;
    return passed;
}
