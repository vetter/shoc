#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <fstream>
#include <iostream>

// When using MPICH and MPICH-derived MPI implementations, there is a
// naming conflict between stdio.h and MPI's C++ binding.
// Since we do not use the C++ MPI binding, we can avoid the ordering
// issue by ignoring the C++ MPI binding headers.
// This #define should be quietly ignored when using other MPI implementations.
#define MPICH_SKIP_MPICXX
#include "mpi.h"


#include "OpenCLDeviceInfo.h"
#include "Event.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "support.h"
#include "Timer.h"

using namespace std;

// Forward Declaration
template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags);

// Template specializations for MPI allreduce call.
template <class T>
inline void globalReduction(T* local_result, T* global_result);

template <>
inline void globalReduction(float* local_result, float* global_result)
{
   MPI_Allreduce(local_result, global_result, 1, MPI_FLOAT,
           MPI_SUM, MPI_COMM_WORLD);
}

template <>
inline void globalReduction(double* local_result, double* global_result)
{
   MPI_Allreduce(local_result, global_result, 1, MPI_DOUBLE,
           MPI_SUM, MPI_COMM_WORLD);
}

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
    // Collect basic MPI information
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Always run single precision test
    // OpenCL doesn't support templated kernels, so we have to use macros
    string spMacros = "-DSINGLE_PRECISION";
    runTest<float>("AllReduce", dev, ctx, queue, resultDB, op, spMacros);

    // If double precision is supported, run the DP test
    if (checkExtension(dev, "cl_khr_fp64"))
    {
        if (rank == 0)
        {
            cout << "DP Supported\n";
        }
        string dpMacros = "-DK_DOUBLE_PRECISION ";
        runTest<double>
        ("AllReduce-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else if (checkExtension(dev, "cl_amd_fp64"))
    {
        if (rank == 0)
        {
            cout << "DP Supported\n";
        }
        string dpMacros = "-DAMD_DOUBLE_PRECISION ";
        runTest<double>
        ("AllReduce-DP", dev, ctx, queue, resultDB, op, dpMacros);
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
            resultDB.AddResult("AllReduce-DP-Kernel" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("AllReduce-DP-Kernel+PCIe" , atts, "GB/s",
                FLT_MAX);
            resultDB.AddResult("AllReduce-DP-MPI_Allreduce" , atts, "GB/s",
                FLT_MAX);
            resultDB.AddResult("AllReduce-DP-Overall" , atts, "GB/s", FLT_MAX);
        }
    }
}

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags)
{
    // Collect basic MPI information
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int err;
    int waitForEvents = 1;

    // Program Setup
    cl_program prog = clCreateProgramWithSource(ctx, 1,
                            &cl_source_reduction, NULL, &err);
    CL_CHECK_ERROR(err);
    if (mpi_rank == 0)
    {
        cout << "Compiling reduction kernel." << endl;
    }

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
    if (getMaxWorkGroupSize(ctx, reduce) == 1)
    {
        nolocal = true;
        localWorkSize = 1;
    }

    int probSizes[4] = { 1, 8, 64, 128 };

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
    if (mpi_rank == 0)
    {
        cout << "Initializing host memory." << endl;
    }

    for(int i=0; i<size; i++)
    {
        h_idata[i] = i % 2; //Fill with some pattern
    }

    // Allocate device memory for input data
    cl_mem d_idata = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes,
            NULL, &err);
    CL_CHECK_ERROR(err);

    int num_blocks;
    if (!nolocal)
    {
        num_blocks = 64;
    }
    else
    {
        num_blocks = 1; // NB: This should only be the case on Apple's CPU
                       // implementation, which is quite restrictive on
                       // work group sizes.
    }

    // Allocate host memory for output
    cl_mem h_o = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof(T)*num_blocks, NULL, &err);
    CL_CHECK_ERROR(err);
    T* h_odata = (T*)clEnqueueMapBuffer(queue, h_o, true,
            CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(T) * num_blocks , 0, NULL, NULL,
            &err);
    CL_CHECK_ERROR(err);

    // Allocate device memory for output
    cl_mem d_odata = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
            num_blocks * sizeof(T), NULL, &err);
    CL_CHECK_ERROR(err);

    // Copy data to GPU
    Event evTransfer("PCIe Transfer");
    err = clEnqueueWriteBuffer(queue, d_idata, true, 0, bytes, h_idata,
            0, NULL, &evTransfer.CLEvent());
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

    if (mpi_rank == 0)
    {
        cout << "Running benchmark." << endl;
    }
    for (int k = 0; k < passes; k++)
    {
        // Synch processes at the start of each test.
        MPI_Barrier(MPI_COMM_WORLD);

        double totalReduceTime = 0.0;
        Event evKernel("reduce kernel");
        for (int m = 0; m < iters; m++)
        {
            if (nolocal)
            {
                err = clEnqueueNDRangeKernel(queue, cpureduce, 1, NULL,
                        &globalWorkSize, &localWorkSize, 0,
                        NULL, &evKernel.CLEvent());
            }
            else
            {
                err = clEnqueueNDRangeKernel(queue, reduce, 1, NULL,
                        &globalWorkSize, &localWorkSize, 0,
                        NULL, &evKernel.CLEvent());
            }
            CL_CHECK_ERROR(err);
            err = clFinish(queue);
            CL_CHECK_ERROR (err);
            evKernel.FillTimingInfo();
            totalReduceTime += (evKernel.SubmitEndRuntime() / 1.e9);
        }

        err = clEnqueueReadBuffer(queue, d_odata, true, 0,
                num_blocks*sizeof(T), h_odata, 0, NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        double totalTransfer = (inputTransfer + evTransfer.StartEndRuntime()) /
                1.e9;

        T local_result = 0.0f, global_result = 0.0f;

        // Start a wallclock timer for MPI
        int TH_global = Timer::Start();

        // Perform reduction of block sums and MPI allreduce call
        for (int m = 0; m < iters; m++)
        {
            local_result = 0.0f;

            for (int i=0; i<num_blocks; i++)
            {
                local_result += h_odata[i];
            }
            global_result = 0.0f;
            globalReduction(&local_result, &global_result);
        }
        double mpi_time = Timer::Stop(TH_global,"global all reduce") / iters;

        // Compute local reference solution
        T cpu_result = reduceCPU<T>(h_idata, size);
        // Use some error threshold for floating point rounding
        double threshold = 1.0e-6;
        T diff = fabs(local_result - cpu_result);

        if (diff > threshold)
        {
            cout << "Error in local reduction detected in rank "
                 << mpi_rank << "\n";
            cout << "Diff: " << diff << endl;
        }

        if (global_result != (mpi_size * local_result))
        {
            cout << "Test Failed, error in global all reduce detected in rank "
                 << mpi_rank << endl;
        }
        else
        {
            if (mpi_rank == 0)
            {
                cout << "Test Passed.\n";
            }
        }
        // Calculate results
        char atts[1024];
        sprintf(atts, "%d_itemsPerRank",size);
        double local_gbytes = (double)(size*sizeof(T))/(1000.*1000.*1000.);
        double global_gbytes = local_gbytes * mpi_size;
        totalReduceTime /= iters; // use average time over the iterations
        resultDB.AddResult(testName+"-Kernel", atts, "GB/s",
            global_gbytes / totalReduceTime);
        resultDB.AddResult(testName+"-Kernel+PCIe", atts, "GB/s",
            global_gbytes / (totalReduceTime + totalTransfer));
        resultDB.AddResult(testName+"-MPI_Allreduce",  atts, "GB/s",
            (sizeof(T)*mpi_size*1.e-9) / (mpi_time));
        resultDB.AddResult(testName+"-Overall", atts, "GB/s",
            global_gbytes / (totalReduceTime + totalTransfer + mpi_time));
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
