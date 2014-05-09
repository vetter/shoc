#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <vector>

#include "OpenCLDeviceInfo.h"
#include "Event.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "support.h"
#include "Timer.h"

using namespace std;

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags);


// ****************************************************************************
// Function: scanCPU
//
// Purpose:
//   Simple cpu scan routine to verify device results
//
// Arguments:
//   data : the input data
//   reference : space for the cpu solution
//   dev_result : result from the device
//   size :
//
// Returns:  nothing, prints relevant info to stdout
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
template <class T>
bool scanCPU(T *data, T* reference, T* dev_result, const size_t size)
{

    bool passed = true;
    T last = 0.0f;

    for (unsigned int i = 0; i < size; ++i)
    {
        reference[i] = data[i] + last;
        last = reference[i];
    }
    for (unsigned int i = 0; i < size; ++i)
    {
        if (reference[i] != dev_result[i])
        {
#ifdef VERBOSE_OUTPUT
            cout << "Mismatch at i: " << i << " ref: " << reference[i]
                 << " dev: " << dev_result[i] << endl;
#endif
            passed = false;
        }
    }
    cout << "Test ";
    if (passed)
        cout << "Passed" << endl;
    else
        cout << "Failed" << endl;
    return passed;
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
    op.addOption("iterations", OPT_INT, "256", "specify scan iterations");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the scan (parallel prefix sum) benchmark
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//   Kyle Spafford, Wed Jun 8 15:20:22 EDT 2011
//   Updating to use non-recursive algorithm
//   Jeremy Meredith, Thu Sep 24 17:30:18 EDT 2009
//   Use implicit include of source file instead of runtime loading.
//
// ****************************************************************************
extern const char *cl_source_scan;

void
RunBenchmark(cl_device_id dev,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    // Always run single precision test
    // OpenCL doesn't support templated kernels, so we have to use macros
    string spMacros = "-DSINGLE_PRECISION";
    runTest<float>("Scan", dev, ctx, queue, resultDB, op, spMacros);

    // If double precision is supported, run the DP test
    if (checkExtension(dev, "cl_khr_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DK_DOUBLE_PRECISION ";
        runTest<double>
        ("Scan-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else if (checkExtension(dev, "cl_amd_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DAMD_DOUBLE_PRECISION ";
        runTest<double>
        ("Scan-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else
    {
        cout << "DP Not Supported\n";
        char atts[1024] = "DP_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (int k = 0; k < passes; k++) {
            resultDB.AddResult("Scan-DP" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("Scan-DP_PCIe" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("Scan-DP_Parity" , atts, "GB/s", FLT_MAX);
        }
    }
}

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags)
{
    int err = 0;

    // Program Setup
    cl_program prog = clCreateProgramWithSource(ctx,
                                                1,
                                                &cl_source_scan,
                                                NULL,
                                                &err);
    CL_CHECK_ERROR(err);

    // Before proceeding, make sure the kernel code compiles and
    // all kernels are valid.
    cout << "Compiling scan kernels." << endl;
    err = clBuildProgram(prog, 1, &dev, compileFlags.c_str(), NULL, NULL);
    CL_CHECK_ERROR(err);

    if (err != CL_SUCCESS)
    {
        char log[5000];
        size_t retsize = 0;
        err = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 5000
                * sizeof(char), log, &retsize);

        CL_CHECK_ERROR(err);
        cout << "Build error." << endl;
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        return;
    }

    // Extract out the 3 kernels
    cl_kernel reduce = clCreateKernel(prog, "reduce", &err);
    CL_CHECK_ERROR(err);

    cl_kernel top_scan = clCreateKernel(prog, "top_scan", &err);
    CL_CHECK_ERROR(err);

    cl_kernel bottom_scan = clCreateKernel(prog, "bottom_scan", &err);
    CL_CHECK_ERROR(err);

    if ( getMaxWorkGroupSize(ctx, reduce)      < 256 ||
         getMaxWorkGroupSize(ctx, top_scan)    < 256 ||
         getMaxWorkGroupSize(ctx, bottom_scan) < 256) {

        cout << "Scan requires a device that supports a work group " <<
          "size of at least 256" << endl;
        char atts[1024] = "GSize_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (int k = 0; k < passes; k++) {
            resultDB.AddResult(testName , atts, "GB/s", FLT_MAX);
            resultDB.AddResult(testName+"_PCIe" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult(testName+"_Parity" , atts, "GB/s", FLT_MAX);
        }
        return;
    }

    // Problem Sizes
    int probSizes[4] = { 1, 8, 32, 64 };
    int size = probSizes[op.getOptionInt("size")-1];

    // Convert to MB
    size = (size * 1024 * 1024) / sizeof(T);

    // Create input data on CPU
    unsigned int bytes = size * sizeof(T);
    T* reference = new T[size];

    // Allocate pinned host memory for input data (h_idata)
    cl_mem h_i = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    T* h_idata = (T*)clEnqueueMapBuffer(queue, h_i, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
    CL_CHECK_ERROR(err);

    // Allocate pinned host memory for output data (h_odata)
    cl_mem h_o = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    T* h_odata = (T*)clEnqueueMapBuffer(queue, h_o, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
    CL_CHECK_ERROR(err);

    // Initialize host memory
    cout << "Initializing host memory." << endl;
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = i % 3; //Fill with some pattern
        h_odata[i] = -1;
    }

    // Allocate device memory for input array
    cl_mem d_idata = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);

    // Allocate device memory for output array
    cl_mem d_odata = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);

    // Number of local work items per group
    const size_t local_wsize  = 256;

    // Number of global work items
    const size_t global_wsize = 16384; // i.e. 64 work groups
    const size_t num_work_groups = global_wsize / local_wsize;

    // Allocate device memory for local work group intermediate sums
    cl_mem d_isums = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
            num_work_groups * sizeof(T), NULL, &err);
    CL_CHECK_ERROR(err);

    // Set the kernel arguments for the reduction kernel
    err = clSetKernelArg(reduce, 0, sizeof(cl_mem), (void*)&d_idata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 1, sizeof(cl_mem), (void*)&d_isums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 2, sizeof(cl_int), (void*)&size);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 3, local_wsize * sizeof(T), NULL);
    CL_CHECK_ERROR(err);

    // Set the kernel arguments for the top-level scan
    err = clSetKernelArg(top_scan, 0, sizeof(cl_mem), (void*)&d_isums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(top_scan, 1, sizeof(cl_int), (void*)&num_work_groups);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(top_scan, 2, local_wsize * 2 * sizeof(T), NULL);
    CL_CHECK_ERROR(err);

    // Set the kernel arguments for the bottom-level scan
    err = clSetKernelArg(bottom_scan, 0, sizeof(cl_mem), (void*)&d_idata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(bottom_scan, 1, sizeof(cl_mem), (void*)&d_isums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(bottom_scan, 2, sizeof(cl_mem), (void*)&d_odata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(bottom_scan, 3, sizeof(cl_int), (void*)&size);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(bottom_scan, 4, local_wsize * 2 * sizeof(T), NULL);
    CL_CHECK_ERROR(err);

    // Copy data to GPU
    cout << "Copying input data to device." << endl;
    Event evTransfer("PCIe transfer");
    err = clEnqueueWriteBuffer(queue, d_idata, true, 0, bytes, h_idata, 0,
            NULL, &evTransfer.CLEvent());
    CL_CHECK_ERROR(err);
    err = clFinish(queue);
    CL_CHECK_ERROR(err);
    evTransfer.FillTimingInfo();
    double inTransferTime = evTransfer.StartEndRuntime();

    // Repeat the test multiplie times to get a good measurement
    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");

    cout << "Running benchmark with size " << size << endl;
    for (int k = 0; k < passes; k++)
    {
        int th = Timer::Start();
        for (int j = 0; j < iters; j++)
        {
            // For scan, we use a reduce-then-scan approach

            // Each thread block gets an equal portion of the
            // input array, and computes the sum.
            err = clEnqueueNDRangeKernel(queue, reduce, 1, NULL,
                        &global_wsize, &local_wsize, 0, NULL, NULL);
            CL_CHECK_ERROR(err);

            // Next, a top-level exclusive scan is performed on the array
            // of block sums
            Event ev_tscan("Top-Level Scan Kernel");
            err = clEnqueueNDRangeKernel(queue, top_scan, 1, NULL,
                        &local_wsize, &local_wsize, 0, NULL, NULL);

            CL_CHECK_ERROR(err);

            // Finally, a bottom-level scan is performed by each block
            // that is seeded with the scanned value in block sums
            err = clEnqueueNDRangeKernel(queue, bottom_scan, 1, NULL,
                        &global_wsize, &local_wsize, 0, NULL, NULL);
            CL_CHECK_ERROR(err);
        }
        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        double totalScanTime = Timer::Stop(th, "total scan time");

        err = clEnqueueReadBuffer(queue, d_odata, true, 0, bytes, h_odata,
                0, NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        double totalTransfer = inTransferTime + evTransfer.StartEndRuntime();
        totalTransfer /= 1.e9; // Convert to seconds

        // If answer is incorrect, stop test and do not report performance
        if (! scanCPU(h_idata, reference, h_odata, size))
        {
            return;
        }

        char atts[1024];
        double avgTime = totalScanTime / (double) iters;
        double gbs = (double) (size * sizeof(T)) / (1000. * 1000. * 1000.);
        sprintf(atts, "%ditems", size);
        resultDB.AddResult(testName, atts, "GB/s", gbs / (avgTime));
        resultDB.AddResult(testName+"_PCIe", atts, "GB/s",
                gbs / (avgTime + totalTransfer));
        resultDB.AddResult(testName+"_Parity", atts, "N",
                totalTransfer / avgTime);
    }

    // Clean up device memory
    err = clReleaseMemObject(d_idata);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_odata);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_isums);
    CL_CHECK_ERROR(err);

    // Clean up pinned host memory
    err = clEnqueueUnmapMemObject(queue, h_i, h_idata, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clEnqueueUnmapMemObject(queue, h_o, h_odata, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_i);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_o);
    CL_CHECK_ERROR(err);

    // Clean up other host memory
    delete[] reference;

    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(reduce);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(top_scan);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(bottom_scan);
    CL_CHECK_ERROR(err);
}

