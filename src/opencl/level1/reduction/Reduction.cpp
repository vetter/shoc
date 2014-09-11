#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <fstream>
#include <iostream>

#include "OpenCLDeviceInfo.h"
#include "Event.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "support.h"

using namespace std;

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags);

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
bool checkResults(T* devResult, T* idata, const int numBlocks, const int size)
{
    T devSum = 0.0;
    for (int i = 0; i < numBlocks; i++)
    {
        devSum += devResult[i];
    }

    T refSum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        refSum += idata[i];
    }

    double threshold = 1.0e-8;
    T diff = fabs(devSum - refSum);

    cout << "Test ";
    if (diff < threshold)
    {
        cout << "Passed\n";
        return true;
    }
    else
    {
        cout << "Failed\nDiff: " << diff << "\n";
        return false;
    }
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
//   Executes the reduction (sum) benchmark
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
//   Jeremy Meredith, Thu Sep 24 17:30:18 EDT 2009
//   Use implicit include of source file instead of
//   runtime loading.
//
// ****************************************************************************
extern const char *cl_source_reduction;

void
RunBenchmark(cl_device_id dev,
        cl_context ctx,
        cl_command_queue queue,
        ResultDatabase &resultDB, OptionParser &op)
{
    // Always run single precision test
    // OpenCL doesn't support templated kernels, so we have to use macros
    string spMacros = "-DSINGLE_PRECISION";
    runTest<float>("Reduction", dev, ctx, queue, resultDB, op, spMacros);

    // If double precision is supported, run the DP test
    if (checkExtension(dev, "cl_khr_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DK_DOUBLE_PRECISION ";
        runTest<double>
        ("Reduction-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else if (checkExtension(dev, "cl_amd_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DAMD_DOUBLE_PRECISION ";
        runTest<double>
        ("Reduction-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else
    {
        cout << "DP Not Supported\n";
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

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags)
{

    int err;
    int waitForEvents = 1;

    // Program Setup
    cl_program prog = clCreateProgramWithSource(ctx, 1,
                            &cl_source_reduction, NULL, &err);
    CL_CHECK_ERROR(err);

    cout << "Compiling reduction kernel." << endl;

    err = clBuildProgram(prog, 1, &dev, compileFlags.c_str(), NULL, NULL);
    CL_CHECK_ERROR(err);

    if (err != 0)
    {
        char log[5000];
        size_t retsize = 0;
        err =  clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                5000*sizeof(char), log, &retsize);

        CL_CHECK_ERROR(err);
        cout << "Build error." << endl;
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        return;
    }

    // Extract out the kernels
    cl_kernel reduce = clCreateKernel(prog, "reduce", &err);
    CL_CHECK_ERROR(err);

    cl_kernel cpureduce = clCreateKernel(prog, "reduceNoLocal", &err);
    CL_CHECK_ERROR(err);

    size_t localWorkSize = 256;
    bool nolocal = false;
    if (getMaxWorkGroupSize(ctx, reduce) == 1) {
        nolocal = true;
        localWorkSize = 1;
    }

    int probSizes[4] = { 1, 8, 32, 64 };

    int size = probSizes[op.getOptionInt("size")-1];
    size = (size * 1024 * 1024) / sizeof(T);

    unsigned int bytes = size * sizeof(T);

    // Allocate pinned host memory for input data
    cl_mem h_i = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    T* h_idata = (T*)clEnqueueMapBuffer(queue, h_i, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
    CL_CHECK_ERROR(err);

    // Initialize host memory
    cout << "Initializing host memory." << endl;
    for(int i=0; i<size; i++)
    {
        h_idata[i] = i % 3; //Fill with some pattern
    }

    // Allocate device memory for input data
    cl_mem d_idata = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes,
            NULL, &err);
    CL_CHECK_ERROR(err);

    int numBlocks;
    if (!nolocal)
    {
        numBlocks = 64;
    }
    else
    {
        numBlocks = 1;
    }

    // Allocate host memory for output
    cl_mem h_o = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(T)*numBlocks, NULL, &err);
    CL_CHECK_ERROR(err);
    T* h_odata = (T*)clEnqueueMapBuffer(queue, h_o, true,
            CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(T) * numBlocks , 0, NULL, NULL,
            &err);
    CL_CHECK_ERROR(err);

    // Allocate device memory for output
    cl_mem d_odata = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
            numBlocks * sizeof(T), NULL, &err);
    CL_CHECK_ERROR(err);

    // Copy data to GPU
    Event evTransfer("PCIe Transfer");
    err = clEnqueueWriteBuffer(queue, d_idata, true, 0, bytes, h_idata,
            0, NULL, &evTransfer.CLEvent());
    CL_CHECK_ERROR(err);
    err = clFinish(queue);
    CL_CHECK_ERROR(err);

    evTransfer.FillTimingInfo();

    double inputTransfer = evTransfer.StartEndRuntime();

    err = clSetKernelArg(reduce, 0, sizeof(cl_mem), (void*)&d_idata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 1, sizeof(cl_mem), (void*)&d_odata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 2,
            localWorkSize * sizeof(T), NULL);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 3, sizeof(cl_int), (void*)&size);
    CL_CHECK_ERROR(err);

    err = clSetKernelArg(cpureduce, 0, sizeof(cl_mem), (void*)&d_idata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpureduce, 1, sizeof(cl_mem), (void*)&d_odata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpureduce, 2, sizeof(cl_int), (void*)&size);
    CL_CHECK_ERROR(err);

    size_t globalWorkSize;
    if (!nolocal)
    {
        globalWorkSize = localWorkSize * 64; // Use 64 work groups
    }
    else
    {
        globalWorkSize = 1;
    }

    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");

    cout << "Running benchmark." << endl;
    for (int k = 0; k < passes; k++)
    {
        double totalReduceTime = 0.0;
        Event evKernel("reduce kernel");
        for (int m = 0; m < iters; m++)
        {
            if (nolocal) {
                err = clEnqueueNDRangeKernel(queue, cpureduce, 1, NULL,
                        &globalWorkSize, &localWorkSize, 0,
                        NULL, &evKernel.CLEvent());

            }
            else {
                err = clEnqueueNDRangeKernel(queue, reduce, 1, NULL,
                        &globalWorkSize, &localWorkSize, 0,
                        NULL, &evKernel.CLEvent());
            }
            CL_CHECK_ERROR(err);
            err = clFinish(queue);
            CL_CHECK_ERROR (err);
            evKernel.FillTimingInfo();
            totalReduceTime += evKernel.SubmitEndRuntime();
        }

        err = clEnqueueReadBuffer(queue, d_odata, true, 0,
                numBlocks*sizeof(T), h_odata, 0, NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        double totalTransfer = (inputTransfer + evTransfer.StartEndRuntime()) /
                1.e9;
        // If result isn't correct, don't report performance
        if (! checkResults(h_odata, h_idata, numBlocks, size))
        {
            return;
        }

        char atts[1024];
        double avgTime = (totalReduceTime / (double)iters) / 1.e9;
        sprintf(atts, "%d_items",size);
        double gbytes = (double)(size*sizeof(T)) / (1000. * 1000. * 1000.);
        resultDB.AddResult(testName, atts, "GB/s", gbytes / avgTime);
        resultDB.AddResult(testName+"_PCIe", atts, "GB/s",
                gbytes / (avgTime + totalTransfer));
        resultDB.AddResult(testName+"_Parity", atts, "N",
                totalTransfer / avgTime);
    }

    err = clEnqueueUnmapMemObject(queue, h_i, h_idata, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clEnqueueUnmapMemObject(queue, h_o, h_odata, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_i);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_o);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_idata);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_odata);
    CL_CHECK_ERROR(err);
    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(reduce);
    CL_CHECK_ERROR(err);
}
