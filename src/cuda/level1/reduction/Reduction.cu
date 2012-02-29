#include "cudacommon.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "reduction_kernel.h"
#include "OptionParser.h"
#include "ResultDatabase.h"

using namespace std;

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

// ****************************************************************************
// Function: reduceCPU
//
// Purpose:
//   Simple cpu reduce routine to verify device results
//
// Arguments:
//   data : the input data
//   size : size of the input data
//
// Returns:  sum of the data
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
template <class T>
T reduceCPU(const T *data, int size)
{
    T sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += data[i];
    }
    return sum;
}

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
void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("iterations", OPT_INT, "256",
                 "specify reduction iterations");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Driver for the reduction benchmark.  Detects double precision capability
//   and calls the RunTest function appropriately
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
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    cout << "Running single precision test" << endl;
    RunTest<float>("Reduction", resultDB, op);

    // Test to see if this device supports double precision
    if ((deviceProp.major == 1 && deviceProp.minor >= 3) ||
               (deviceProp.major >= 2))
    {
        cout << "Running double precision test" << endl;
        RunTest<double>("Reduction-DP", resultDB, op);
    } else {
        cout << "Skipping double precision test" << endl;
        char atts[1024] = "DP_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (int k = 0; k < passes; k++) {
            resultDB.AddResult("Reduction-DP" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("Reduction-DP_PCIe" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("Reduction-DP_Parity" , atts, "GB/s", FLT_MAX);
        }
    }

}
// ****************************************************************************
// Function: RunTest
//
// Purpose:
//   Primary method for the reduction benchmark
//
// Arguments:
//   testName: the name of the test currently being executed (specifying SP or
//             DP)
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
template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{
    int prob_sizes[4] = { 1, 8, 32, 64 };

    int size = prob_sizes[op.getOptionInt("size")-1];
    size = (size * 1024 * 1024) / sizeof(T);

    T* h_idata;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&h_idata, size * sizeof(T)));

    // Initialize host memory
    cout << "Initializing host memory." << endl;
    for(int i = 0; i < size; i++)
    {
        h_idata[i] = i % 3; //Fill with some pattern
    }

    // allocate device memory
    T* d_idata;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_idata, size * sizeof(T)));

    int num_threads = 256; // NB: Update template to kernel launch
                           // if this is changed
    int num_blocks = 64;
    int smem_size = sizeof(T) * num_threads;
    // allocate mem for the result on host side
    T* h_odata;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&h_odata, num_blocks * sizeof(T)));

    T* d_odata;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_odata, num_blocks * sizeof(T)));

    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");

    cout << "Running benchmark." << endl;
    for (int k=0; k<passes; k++)
    {
        // Copy data to GPU
        cudaEvent_t start, stop;
        CUDA_SAFE_CALL(cudaEventCreate(&start));
        CUDA_SAFE_CALL(cudaEventCreate(&stop));
        CUDA_SAFE_CALL(cudaEventRecord(start, 0));
        CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, size*sizeof(T),
                cudaMemcpyHostToDevice));
        cudaEventRecord(stop, 0);
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));

        // Get elapsed time
        float transfer_time = 0.0f;
        cudaEventElapsedTime(&transfer_time, start, stop);
        transfer_time *= 1.e-3;

        // Execute kernel
        cudaEventRecord(start, 0);
        for (int m = 0; m < iters; m++)
        {
            reduce<T,256><<<num_blocks,num_threads, smem_size>>>
                (d_idata, d_odata, size);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        // Get kernel time
        float totalReduceTime;
        cudaEventElapsedTime(&totalReduceTime, start, stop);
        double avg_time = totalReduceTime / (double)iters;
        avg_time *= 1.e-3; // convert to seconds

        // Copy back to host
        cudaEventRecord(start, 0);
        CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata,
                num_blocks*sizeof(T), cudaMemcpyDeviceToHost));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float output_time;
        cudaEventElapsedTime(&output_time, start, stop);
        output_time *= 1.e-3;
        transfer_time += output_time;

        T dev_result = 0;
        for (int i=0; i<num_blocks; i++)
        {
            dev_result += h_odata[i];
        }

        // compute reference solution
        T cpu_result = reduceCPU<T>(h_idata, size);
        double threshold = 1.0e-6;
        T diff = fabs(dev_result - cpu_result);

        cout << "Test ";
        if (diff < threshold)
            cout << "Passed";
        else
        {
            cout << "FAILED\n";
            cout << "Diff: " << diff;
            return; // (don't report erroneous results)
        }
        cout << endl;

        // Calculate results
        char atts[1024];
        sprintf(atts, "%d_items",size);
        double gbytes = (double)(size*sizeof(T))/(1000.*1000.*1000.);
        resultDB.AddResult(testName, atts, "GB/s", gbytes / avg_time);
        resultDB.AddResult(testName+"_PCIe", atts, "GB/s", gbytes /
                (avg_time + transfer_time));
        resultDB.AddResult(testName+"_Parity", atts, "N",
                transfer_time / avg_time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    CUDA_SAFE_CALL(cudaFreeHost(h_idata));
    CUDA_SAFE_CALL(cudaFreeHost(h_odata));
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));
}
