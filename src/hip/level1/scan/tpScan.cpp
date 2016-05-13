#include "cudacommon.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <iostream>
#include <vector>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"
#include "TPScan.h"

using namespace std;

// Forward declarations
template <class T, class vecT>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

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
    op.addOption("iterations", OPT_INT, "256", "specify scan iterations");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the scan (parallel prefix sum) benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
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
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    // Always do the single precision test
    RunTest<float,float4>("TPScan-SP", resultDB, op);

    // Test to see if this device supports double precision
    if ((deviceProp.major == 1 && deviceProp.minor >= 3) ||
               (deviceProp.major >= 2))
    {
        if (rank == 0)
        {
            cout << "Starting double precision tests\n";
        }
        RunTest<double,double4>("TPScan-DP", resultDB, op);
    }
    else
    {
        char atts[1024] = "DP_Not_Supported";
        cout << "Warning, rank " << rank << "'s device does not support DP\n";
        // ResultDB requires every rank to report something. If this rank
        // doesn't support DP, submit FLT_MAX (this is handled as no result by
        // ResultDB.
        int passes = op.getOptionInt("passes");
        for (int k = 0; k < passes; k++)
        {
            resultDB.AddResult("TPScan-DP-Kernel" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("TPScan-DP-Kernel+PCIe" , atts, "GB/s",
                FLT_MAX);
            resultDB.AddResult("TPScan-DP-MPI_ExScan" , atts, "GB/s",
                FLT_MAX);
            resultDB.AddResult("TPScan-DP-Overall" , atts, "GB/s", FLT_MAX);
        }
    }
}

template <class T, class vecT>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int prob_sizes[4] = { 1, 8, 32, 64 };
    int size_class = op.getOptionInt("size");
    assert(size_class > 0 && size_class < 5);
    int size = prob_sizes[size_class-1];

    // Convert to MB
    size = (size * 1024 * 1024) / sizeof(T);
    // create input data on CPU
    unsigned int bytes = size * sizeof(T);

    // Thread configuration
    int num_threads = 256;
    int num_blocks = 64;
    int smem_size = num_threads * sizeof(T);

    // Allocate Host Memory
    T* h_idata, *h_block_sums, *h_odata;
    CUDA_SAFE_CALL(cudaMallocHost((void**) &h_idata, bytes));
    CUDA_SAFE_CALL(cudaMallocHost((void**) &h_block_sums,
                num_blocks * sizeof(T)));
    CUDA_SAFE_CALL(cudaMallocHost((void**) &h_odata, bytes));

    // Allocate Device Data
    T* d_idata, *d_block_sums, *d_odata;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_idata, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_block_sums, num_blocks * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_odata, bytes));

    // Initialize host memory
    if (mpi_rank == 0)
    {
        cout << "Initializing host memory." << endl;
    }

    for (int i = 0; i < size; i++)
    {
        h_idata[i] = i % 2; // Fill with some pattern
        h_odata[i] = -1;
    }

    int passes = op.getOptionInt("passes");
    int iters = op.getOptionInt("iterations");
    cudaEvent_t start, stop;

    for (int k = 0; k < passes; k++)
    {
        // Timing variables
        double kernel_time=0., pcie_time=0., mpi_time=0.;

        // Copy data to GPU
        CUDA_SAFE_CALL(cudaEventCreate(&start));
        CUDA_SAFE_CALL(cudaEventCreate(&stop));
        MPI_Barrier(MPI_COMM_WORLD); // Sync processes at beginning of pass

        CUDA_SAFE_CALL(cudaEventRecord(start, 0));
        for (int m = 0; m < iters; m++)
        {
            CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, bytes,
                           cudaMemcpyHostToDevice));
        }
        cudaEventRecord(stop, 0);
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));

        // Get elapsed time of input PCIe transfer
        float temp;
        cudaEventElapsedTime(&temp, start, stop);
        pcie_time += (temp / (double)iters)* 1.e-3;

        // This code uses a reduce-then-scan strategy.
        // The major steps of the algorithm are:
        // 1. Local reduction on a node
        // 2. Global exclusive scan of the reduction values
        // 3. Local inclusive scan, seeded with the node's result
        //    from the global exclusive scan
        CUDA_SAFE_CALL(cudaEventRecord(start, 0));
        for (int j = 0; j < iters; j++)
        {
            LaunchReduceKernel<T>( num_blocks,
                                   num_threads,
                                   smem_size,
                                   d_idata,
                                   d_block_sums,
                                   size );
        }
        CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&temp, start, stop);
        kernel_time += (temp / (float)iters) * 1.e-3;

        // Next step is to copy the reduced blocks back to the host,
        // sum them, and perform the MPI exlcusive (top level) scan.
        CUDA_SAFE_CALL(cudaEventRecord(start, 0));
        for (int m = 0; m < iters; m++)
        {
            // Copy back to host
            CUDA_SAFE_CALL(cudaMemcpy(h_block_sums, d_block_sums,
                num_blocks*sizeof(T), cudaMemcpyDeviceToHost));
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp, start, stop);
        pcie_time += (temp / (double)iters) * 1.e-3;

        int globscan_th = Timer::Start();
        T reduced=0., scanned=0.;
        // To get the true sum for this node, we have to add up
        // the block sums before MPI scanning.
        for (int i = 0; i < num_blocks; i++)
        {
            reduced += h_block_sums[i];
        }

        // Next step is an exclusive scan across MPI ranks.
        // Then a local scan seeded with the result from MPI.
        for (int j = 0; j < iters; j++)
        {
            globalExscan(&reduced, &scanned);
        }
        mpi_time += Timer::Stop(globscan_th, "Global Scan") / iters;

        // Now, scanned contains all the information we need from other nodes
        // Next step is to perform the local top level (i.e. across blocks) scan,
        // but seed it with the "scanned", the sum of elems on all lower ranks.
        h_block_sums[0] += scanned;
        CUDA_SAFE_CALL(cudaEventRecord(start, 0));
        CUDA_SAFE_CALL(cudaMemcpy(d_block_sums, h_block_sums,
                sizeof(T), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&temp, start, stop);
        pcie_time +=  temp * 1.e-3;

        // Device block sums has been seeded, perform the top level scan,
        // then the bottom level scan.
        CUDA_SAFE_CALL(cudaEventRecord(start, 0));
        LaunchTopScanKernel( 1,
                             num_threads,
                             smem_size*2,
                             d_block_sums,
                             num_blocks );

        LaunchBottomScanKernel<T, vecT, 256>( num_blocks,
                                              num_threads,
                                              smem_size * 2,
                                              d_idata,
                                              d_odata,
                                              d_block_sums,
                                              size );
        CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&temp, start, stop);
        kernel_time += temp * 1.e-3;

        // Lightweight correctness check -- won't apply
        // if data is not initialized to i%2 above
        if (mpi_rank == mpi_size-1)
        {
            CUDA_SAFE_CALL(cudaMemcpy(&(h_odata[size-1]),
                                      &(d_odata[size-1]),
                sizeof(T), cudaMemcpyDeviceToHost));

            if (h_odata[size-1] != (mpi_size * size) / 2)
            {
                cout << "Test Failed\n";
            }
            else
            {
                cout << "Test Passed\n";
            }
        }

        char atts[1024];
        sprintf(atts, "%d items", size);
        double global_gb = (double)(mpi_size * size * sizeof(T)) / 1e9;

        resultDB.AddResult(testName+"-Kernel" , atts, "GB/s",
                global_gb / kernel_time);
        resultDB.AddResult(testName+"-Kernel+PCIe" , atts, "GB/s",
                global_gb / (kernel_time + pcie_time));
        resultDB.AddResult(testName+"-MPI_ExScan" , atts, "GB/s",
                (mpi_size * sizeof(T) *1e-9) / mpi_time);
        resultDB.AddResult(testName+"-Overall" , atts, "GB/s",
                global_gb / (kernel_time + pcie_time + mpi_time));
    }

    // Clean up host data
    CUDA_SAFE_CALL(cudaFreeHost(h_idata));
    CUDA_SAFE_CALL(cudaFreeHost(h_odata));
    CUDA_SAFE_CALL(cudaFreeHost(h_block_sums));
    // Clean up device data
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));
    CUDA_SAFE_CALL(cudaFree(d_block_sums));
    // Clean up events used in timing
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
}

