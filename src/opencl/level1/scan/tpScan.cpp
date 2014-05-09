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
#include "TPScan.h"

using namespace std;

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags);

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
    ;
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
    // Collect basic MPI information
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Always run single precision test
    // OpenCL doesn't support templated kernels, so we have to use macros
    string spMacros = "-DSINGLE_PRECISION";
    runTest<float>
        ("TPScan-SP", dev, ctx, queue, resultDB, op, spMacros);

    // If double precision is supported, run the DP test
    if (checkExtension(dev, "cl_khr_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DK_DOUBLE_PRECISION ";
        runTest<double>
        ("TPScan-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else if (checkExtension(dev, "cl_amd_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DAMD_DOUBLE_PRECISION ";
        runTest<double>
        ("TPScan-DP", dev, ctx, queue, resultDB, op, dpMacros);
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

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags)
{
    int err = 0;

    // Collect basic MPI information
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Program Setup
    cl_program prog = clCreateProgramWithSource(ctx,
                                                1,
                                                &cl_source_scan,
                                                NULL,
                                                &err);
    CL_CHECK_ERROR(err);

    // Before proceeding, make sure the kernel code compiles and
    // all kernels are valid.
    if (mpi_rank == 0)
    {
        cout << "Compiling scan kernels." << endl;
    }
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

    // If the device doesn't support at least 256 work items in a
    // group, use a different kernel (TODO)
    if (getMaxWorkGroupSize(dev) < 256)
    {
        cout << "Scan requires work group size of at least 256" << endl;
        char atts[1024] = "GSize_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (int k = 0; k < passes; k++)
        {
            resultDB.AddResult(testName+"-Kernel" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult(testName+"-Kernel+PCIe" , atts, "GB/s",
                FLT_MAX);
            resultDB.AddResult(testName+"-MPI_ExScan" , atts, "GB/s",
                FLT_MAX);
            resultDB.AddResult(testName+"-Overall" , atts, "GB/s", FLT_MAX);
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
    if (mpi_rank == 0)
    {
        cout << "Initializing host memory." << endl;
    }
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = i % 2; //Fill with some pattern
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

    // Number of local work groups and total work items
    const size_t num_work_groups = 64;
    const size_t global_wsize = local_wsize * num_work_groups;

    // Allocate device memory for local work group intermediate sums
    cl_mem d_isums = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
            num_work_groups * sizeof(T), NULL, &err);
    CL_CHECK_ERROR(err);

    // Allocate pinned host memory for intermediate block sums (h_isums)
    cl_mem h_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        num_work_groups * sizeof(T), NULL, &err);
    CL_CHECK_ERROR(err);
    T* h_isums = (T*)clEnqueueMapBuffer(queue, h_b, true,
        CL_MAP_READ|CL_MAP_WRITE, 0, num_work_groups * sizeof(T),
        0, NULL, NULL, &err);
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

    // Repeat the test multiple times to get a good measurement
    int passes = op.getOptionInt("passes");

    if (mpi_rank == 0)
    {
        cout << "Running benchmark with size " << size << endl;
    }
    for (int k = 0; k < passes; k++)
    {
        // Timing variables
        double pcie_time=0., kernel_time=0., mpi_time=0.;

        // Copy data to GPU
        Event evTransfer("PCIe transfer");
        double time_temp = 0.;
        err = clEnqueueWriteBuffer(queue, d_idata, true, 0, bytes, h_idata, 0,
                NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        pcie_time += (double)evTransfer.StartEndRuntime() / 1e9;

        // This code uses a reduce-then-scan strategy.
        // The major steps of the algorithm are:
        // 1. Local reduction on a node
        // 2. Global exclusive scan of the reduction values
        // 3. Local inclusive scan, seeded with the node's result
        //    from the global exclusive scan
        Event ev_reduce("Reduction Kernel");
        err = clEnqueueNDRangeKernel(queue, reduce, 1, NULL,
                    &global_wsize, &local_wsize, 0, NULL,
                    &ev_reduce.CLEvent());
        err = clFinish(queue);
        ev_reduce.FillTimingInfo();
        kernel_time += (double)ev_reduce.StartEndRuntime() * 1e-9;

        // Next step is to copy the reduced blocks back to the host,
        // sum them, and perform the MPI exlcusive (top level) scan.
        err = clEnqueueReadBuffer(queue, d_isums, true, 0,
                num_work_groups*sizeof(T), h_isums, 0,
                NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        pcie_time += (double)evTransfer.StartEndRuntime() * 1e-9;

        // Start the timer for MPI Scan
        int globscan_th = Timer::Start();
        T reduced=0., scanned=0.;

        // To get the true sum for this node, we have to add up
        // the block sums before MPI scanning.
        for (int i = 0; i < num_work_groups; i++)
        {
            reduced += h_isums[i];
        }

        // Next step is an exclusive scan across MPI ranks.
        // Then a local scan seeded with the result from MPI.
        globalExscan(&reduced, &scanned);
        mpi_time += Timer::Stop(globscan_th, "Global Scan");

        // Now, scanned contains all the information we need from other nodes
        // Next step is to perform the local top level (i.e. across blocks) scan,
        // but seed it with the "scanned", the sum of elems on all lower ranks.
        h_isums[0] += scanned;

        err = clEnqueueWriteBuffer(queue, d_isums, true, 0, sizeof(T), h_isums, 0,
                NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        pcie_time += (double)evTransfer.StartEndRuntime() * 1e-9;

        Event ev_scan("Scan Kernel");
        err = clEnqueueNDRangeKernel(queue, top_scan, 1, NULL,
                &local_wsize, &local_wsize, 0, NULL, &ev_scan.CLEvent());
        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        ev_scan.FillTimingInfo();
        kernel_time += ((double)ev_scan.StartEndRuntime() * 1.e-9);

        // Finally, a bottom-level scan is performed by each block
        // that is seeded with the scanned value in block sums
        err = clEnqueueNDRangeKernel(queue, bottom_scan, 1, NULL,
                    &global_wsize, &local_wsize, 0, NULL,
                    &ev_scan.CLEvent());
        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        ev_scan.FillTimingInfo();
        kernel_time += ((double)ev_scan.StartEndRuntime() * 1.e-9);

        // Read data back for correctness check
        err = clEnqueueReadBuffer(queue, d_odata, true, 0, bytes, h_odata,
                0, NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);

        // Lightweight correctness check -- won't apply
        // if data is not initialized to i%2 above
        if (mpi_rank == mpi_size-1)
        {
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
        double global_gb = (double)(mpi_size * size * sizeof(T)) /
            (1000. * 1000. * 1000.);

        resultDB.AddResult(testName+"-Kernel" , atts, "GB/s",
                global_gb / kernel_time);
        resultDB.AddResult(testName+"-Kernel+PCIe" , atts, "GB/s",
                global_gb / (kernel_time + pcie_time));
        resultDB.AddResult(testName+"-MPI_ExScan" , atts, "GB/s",
                (mpi_size * sizeof(T) *1e-9) / mpi_time);
        resultDB.AddResult(testName+"-Overall" , atts, "GB/s",
                global_gb / (kernel_time + pcie_time + mpi_time));
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
    err = clEnqueueUnmapMemObject(queue, h_b, h_isums, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_i);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_o);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_b);
    CL_CHECK_ERROR(err);

    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(reduce);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(top_scan);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(bottom_scan);
    CL_CHECK_ERROR(err);
}


