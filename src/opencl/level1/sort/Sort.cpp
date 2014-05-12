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
bool verifySort(unsigned int *keys, const size_t size)
{
    bool passed = true;

    for (unsigned int i = 0; i < size - 1; i++)
    {
        if (keys[i] > keys[i + 1])
        {
            passed = false;
#ifdef VERBOSE_OUTPUT
            cout << "Idx: " << i;
            cout << " Key: " << keys[i] << "\n";
#endif
        }
    }
    cout << "Test ";
    if (passed)
        cout << "Passed" << endl;
    else
        cout << "---FAILED---" << endl;

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
    ; // No specific options for this benchmark
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the radix sorting benchmark
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
extern const char *cl_source_sort;

void
RunBenchmark(cl_device_id dev,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    // Execute the test using 32-bit keys only
    runTest<unsigned int>("Sort-Rate", dev, ctx, queue, resultDB, op, " ");
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
                                                &cl_source_sort,
                                                NULL,
                                                &err);
    CL_CHECK_ERROR(err);

    // Before proceeding, make sure the kernel code compiles and
    // all kernels are valid.
    cout << "Compiling sort kernels." << endl;
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
    // Note that these kernels are analogs of those in use for
    // scan, but have had "visiting" logic added to them
    // as described by Merrill et al. See
    // http://www.cs.virginia.edu/~dgm4d/
    cl_kernel reduce = clCreateKernel(prog, "reduce", &err);
    CL_CHECK_ERROR(err);

    cl_kernel top_scan = clCreateKernel(prog, "top_scan", &err);
    CL_CHECK_ERROR(err);

    cl_kernel bottom_scan = clCreateKernel(prog, "bottom_scan", &err);
    CL_CHECK_ERROR(err);

    // If the device doesn't support at least 256 work items in a
    // group, use a different kernel (TODO)
    if ( getMaxWorkGroupSize(ctx, reduce)      < 256 ||
         getMaxWorkGroupSize(ctx, top_scan)    < 256 ||
         getMaxWorkGroupSize(ctx, bottom_scan) < 256) {
        cout << "Sort requires a device that supports a work group size " <<
          "of at least 256" << endl;
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

    // Convert to MiB
    size = (size * 1024 * 1024) / sizeof(T);

    // Create input data on CPU
    unsigned int bytes = size * sizeof(T);

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
        h_idata[i] = i % 16; // Fill with some pattern
        h_odata[i] = -1;
    }

    // The radix width in bits
    const int radix_width = 4; // Changing this requires major kernel updates
    const int num_digits = (int)pow((double)2, radix_width); // n possible digits

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
            num_work_groups * num_digits * sizeof(T), NULL, &err);
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

    cout << "Running benchmark with size " << size << endl;
    for (int k = 0; k < passes; k++)
    {
        int th = Timer::Start();
        // Assuming an 8 bit byte.
        for (int shift = 0; shift < sizeof(T)*8; shift += radix_width)
        {
            // Like scan, we use a reduce-then-scan approach

            // But before proceeding, update the shift appropriately
            // for each kernel. This is how many bits to shift to the
            // right used in binning.
            err = clSetKernelArg(reduce, 4, sizeof(cl_int), (void*)&shift);
            CL_CHECK_ERROR(err);

            err = clSetKernelArg(bottom_scan, 5, sizeof(cl_int), (void*)&shift);
            CL_CHECK_ERROR(err);

            // Also, the sort is not in place, so swap the input and output
            // buffers on each pass.
            bool even = ((shift / radix_width) % 2 == 0) ? true : false;

            if (even)
            {
                // Set the kernel arguments for the reduction kernel
                err = clSetKernelArg(reduce, 0, sizeof(cl_mem),
                        (void*)&d_idata);
                CL_CHECK_ERROR(err);
                // Set the kernel arguments for the bottom-level scan
                err = clSetKernelArg(bottom_scan, 0, sizeof(cl_mem),
                        (void*)&d_idata);
                CL_CHECK_ERROR(err);
                err = clSetKernelArg(bottom_scan, 2, sizeof(cl_mem),
                        (void*)&d_odata);
                CL_CHECK_ERROR(err);
            }
            else // i.e. odd pass
            {
                // Set the kernel arguments for the reduction kernel
                err = clSetKernelArg(reduce, 0, sizeof(cl_mem),
                        (void*)&d_odata);
                CL_CHECK_ERROR(err);
                // Set the kernel arguments for the bottom-level scan
                err = clSetKernelArg(bottom_scan, 0, sizeof(cl_mem),
                        (void*)&d_odata);
                CL_CHECK_ERROR(err);
                err = clSetKernelArg(bottom_scan, 2, sizeof(cl_mem),
                        (void*)&d_idata);
                CL_CHECK_ERROR(err);
            }

            // Each thread block gets an equal portion of the
            // input array, and computes occurrences of each digit.
            err = clEnqueueNDRangeKernel(queue, reduce, 1, NULL,
                        &global_wsize, &local_wsize, 0, NULL, NULL);

            // Next, a top-level exclusive scan is performed on the
            // per block histograms.  This is done by a single
            // work group (note global size here is the same as local).
            err = clEnqueueNDRangeKernel(queue, top_scan, 1, NULL,
                        &local_wsize, &local_wsize, 0, NULL, NULL);

            // Finally, a bottom-level scan is performed by each block
            // that is seeded with the scanned histograms which rebins,
            // locally scans, then scatters keys to global memory
            err = clEnqueueNDRangeKernel(queue, bottom_scan, 1, NULL,
                        &global_wsize, &local_wsize, 0, NULL, NULL);
        }
        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        double total_sort = Timer::Stop(th, "total sort time");

        err = clEnqueueReadBuffer(queue, d_idata, true, 0, bytes, h_odata,
                0, NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        double totalTransfer = inTransferTime + evTransfer.StartEndRuntime();
        totalTransfer /= 1.e9; // Convert to seconds

        // If answer is incorrect, stop test and do not report performance
        if (! verifySort(h_odata, size))
        {
            return;
        }

        char atts[1024];
        double avgTime = total_sort;
        double gbs = (double) (size * sizeof(T)) / (1000. * 1000. * 1000.);
        sprintf(atts, "%d_items", size);
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

    // Clean up program and kernel objects
    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(reduce);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(top_scan);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(bottom_scan);
    CL_CHECK_ERROR(err);
}
