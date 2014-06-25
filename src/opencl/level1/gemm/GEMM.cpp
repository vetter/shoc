#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cassert>
#include <iostream>
#include <sstream>

#include "OpenCLDeviceInfo.h"
#include "Event.h"
#include "OptionParser.h"
#include "support.h"
#include "ResultDatabase.h"

using namespace std;

#define CL_BAIL_ON_ERROR(err) \
{                             \
    CL_CHECK_ERROR(err);      \
    if (err != CL_SUCCESS)    \
        return;               \
}

// Forward declaration
template <class T> inline std::string toString (const T& t){
    std::stringstream ss;
    ss << t;
    return ss.str();
}

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
    op.addOption("KiB", OPT_INT, "0", "data size (in Kibibytes)");
}


// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Benchmarks the GEMM codes
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: August 26, 2009
//
// Modifications:
//   Jeremy Meredith, Thu Sep 24 17:30:18 EDT 2009
//   Use implicit include of source file instead of runtime loading.
//
//   Jeremy Meredith, Thu Aug 19 13:59:09 EDT 2010
//   Added transfer vs computation equivalence calculation.
//
//   Jeremy Meredith, Thu Aug 19 14:16:49 EDT 2010
//   Use pinned memory for better PCIe speeds.
//
// ****************************************************************************
extern const char *cl_source_gemmN;

void
RunBenchmark(cl_device_id dev,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    // Always run single precision test
    // OpenCL doesn't support templated kernels, so we have to use macros
    runTest<float>("SGEMM", dev, ctx, queue, resultDB, op,
            "-DSINGLE_PRECISION");

    // If double precision is supported, run the DP test
    if (checkExtension(dev, "cl_khr_fp64"))
    {
        cout << "DP Supported\n";
        runTest<double>("DGEMM", dev, ctx, queue, resultDB, op,
                "-DK_DOUBLE_PRECISION ");
    }
    else if (checkExtension(dev, "cl_amd_fp64"))
    {
        cout << "DP Supported\n";
        runTest<double>("DGEMM", dev, ctx, queue, resultDB, op,
                "-DAMD_DOUBLE_PRECISION ");
    }
    else
    {
        cout << "DP Not Supported\n";
        char atts[1024] = "DP_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (; passes > 0; --passes) {
            for (int i = 0; i < 2; i++) {
                const char transb = i ? 'T' : 'N';
                string testName="DGEMM";
                resultDB.AddResult(testName+"-"+transb, atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_PCIe", atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_Parity", atts, "N", FLT_MAX);
            }
        }
    }
}

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags)
{

    int N;
    if (op.getOptionInt("KiB") == 0)
    {
        int probSizes[4] = { 1, 4, 8, 16 };
        N = probSizes[op.getOptionInt("size")-1] * 1024 / sizeof(T);
    } else {
        N = op.getOptionInt("KiB") * 1024 / sizeof(T);
    }

    cl_int err;
    int waitForEvents = 1;
    size_t m = N, n = N, k = N;
    size_t lda, ldb, ldc;
    const T alpha = 1;
    const T beta = -1;
    int i, j;

    lda = ldb = ldc = N;

    cl_uint numDimensions = 0;
    clGetDeviceInfo (dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                             sizeof(cl_uint), &numDimensions, NULL);
    size_t *maxWorkSizes = new size_t[numDimensions];
    clGetDeviceInfo (dev, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                       sizeof(size_t)*numDimensions, maxWorkSizes, NULL);

    if (numDimensions<2 || maxWorkSizes[0]<16 || maxWorkSizes[1] < 4)
    {
        cout << "SGEMM needs a 2-dimensional work group size of at least {16,4}." << endl;
        int passes = op.getOptionInt("passes");
        char atts[1024] = "GSize_Not_Supported";
        for (; passes > 0; --passes) {
            for (i = 0; i < 2; i++) {
                const char transb = i ? 'T' : 'N';
                resultDB.AddResult(testName+"-"+transb, atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_PCIe", atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_Parity", atts, "N", FLT_MAX);
            }
        }
        return;
    }

    size_t localWorkSize[2] = {16,4};


    // Create program object
    cl_program prog = clCreateProgramWithSource(ctx, 1,
                                 &cl_source_gemmN, NULL, &err);
    CL_CHECK_ERROR(err);

    string flags = compileFlags + " -cl-mad-enable";
    err = clBuildProgram(prog, 0, NULL, flags.c_str(), NULL,
            NULL);
    CL_CHECK_ERROR(err);

    // If compilation fails, print error messages and return
    if (err != CL_SUCCESS) {
        char log[5000];
        size_t retsize = 0;
        err =  clGetProgramBuildInfo (prog, dev, CL_PROGRAM_BUILD_LOG,
                5000*sizeof(char),  log, &retsize);

        CL_CHECK_ERROR(err);
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        exit(-1);
    }

    // Generate the kernel objects
    cl_kernel sgemmNN = clCreateKernel(prog, "sgemmNN", &err);
    CL_CHECK_ERROR(err);

    cl_kernel sgemmNT = clCreateKernel(prog, "sgemmNT", &err);
    CL_CHECK_ERROR(err);

    // Allocate memory for the matrices
    T *A, *B, *C;
    cl_mem Aobj, Bobj, Cobj;
    if (true) // pinned
    {
        Aobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
			       sizeof(T)*N*N, NULL, &err);
        CL_CHECK_ERROR(err);
        A =(T*)clEnqueueMapBuffer(queue,Aobj,true,CL_MAP_READ|CL_MAP_WRITE,
				       0,sizeof(T)*N*N,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);

        Bobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
			       sizeof(T)*N*N, NULL, &err);
        CL_CHECK_ERROR(err);
        B =(T*)clEnqueueMapBuffer(queue,Bobj,true,CL_MAP_READ|CL_MAP_WRITE,
				       0,sizeof(T)*N*N,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);

        Cobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
			       sizeof(T)*N*N, NULL, &err);
        CL_CHECK_ERROR(err);
        C =(T*)clEnqueueMapBuffer(queue,Cobj,true,CL_MAP_READ|CL_MAP_WRITE,
				       0,sizeof(T)*N*N,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);
    }
    else
    {
	A = (T*)malloc( N*N*sizeof( T ) );
	B = (T*)malloc( N*N*sizeof( T ) );
	C = (T*)malloc( N*N*sizeof( T ) );
    }

    // Initialize inputs
    srand48(13579862);
    for(i=0; i<m; ++i){
        for(j=0; j<k; ++j){
            A[i*k+j] = (T)(0.5 + drand48()*1.5);
        }
    }

    for(i=0; i<k; ++i){
        for(j=0; j<n; ++j){
            B[i*n+j] = (T)(0.5 + drand48()*1.5);
        }
    }

    for(i=0; i<m; ++i){
        for(j=0; j<n; ++j){
            C[i*n+j] = 0.0;
        }
    }

    // Pass A and B to the GPU and create a GPU buffer for C
    cl_mem Agpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 m*k * sizeof(T), NULL, &err);
    CL_BAIL_ON_ERROR(err);
    cl_mem Bgpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 k*n * sizeof(T), NULL, &err);
    CL_BAIL_ON_ERROR(err);
    cl_mem Cgpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 m*n * sizeof(T), NULL, &err);
    CL_BAIL_ON_ERROR(err);


    // Set arguments to the sgemmNN kernel
    err = clSetKernelArg(sgemmNN, 0, sizeof(cl_mem), (void*)&Agpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 1, sizeof(int), (void*)&lda);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 2, sizeof(cl_mem), (void*)&Bgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 3, sizeof(int), (void*)&ldb);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 4, sizeof(cl_mem), (void*)&Cgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 5, sizeof(int), (void*)&ldc);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 6, sizeof(int), (void*)&k);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 7, sizeof(T), (void*)&alpha);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 8, sizeof(T), (void*)&beta);
    CL_BAIL_ON_ERROR(err);

    // Pass arguments to the sgemmNT kernel
    err = clSetKernelArg(sgemmNT, 0, sizeof(cl_mem), (void*)&Agpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 1, sizeof(int), (void*)&lda);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 2, sizeof(cl_mem), (void*)&Bgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 3, sizeof(int), (void*)&ldb);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 4, sizeof(cl_mem), (void*)&Cgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 5, sizeof(int), (void*)&ldc);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 6, sizeof(int), (void*)&k);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 7, sizeof(T), (void*)&alpha);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 8, sizeof(T), (void*)&beta);
    CL_BAIL_ON_ERROR(err);

    const size_t globalWorkSize[2] = {m/4,n/4};

    int passes = op.getOptionInt("passes");

    // Run NN
    for (int i = 0; i < passes; i++) {
        Event evDownload1("Download A");
        Event evUpload("Upload");
        Event evNN("sgemmNN");

        err = clEnqueueWriteBuffer(queue, Agpu, CL_TRUE, 0, m*n*sizeof(T),
                A, 0, NULL, &evDownload1.CLEvent());
        err = clEnqueueWriteBuffer(queue, Bgpu, CL_TRUE, 0, m*n*sizeof(T),
                B, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                C, 0, NULL, NULL);

        // Wait until data transfers finish
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        //Launch Kernels
        err = clEnqueueNDRangeKernel(queue, sgemmNN, 2, NULL, globalWorkSize,
                                     localWorkSize, 0, NULL, &evNN.CLEvent());
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        err = clEnqueueReadBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                C, 0, NULL, &evUpload.CLEvent());
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        evNN.FillTimingInfo();
        evDownload1.FillTimingInfo();
        evUpload.FillTimingInfo();

        double user_wait_time = 0.0;
        double gemm_pure_time = 0.0;

        gemm_pure_time = evNN.SubmitEndRuntime();
        user_wait_time = evUpload.EndTime() - evDownload1.QueuedTime();
        double transfer_time = user_wait_time - gemm_pure_time;
        double flops = 2.0*(double)N*N*N;
        resultDB.AddResult(testName+"-N", toString(N), "GFLOPS",
                flops / gemm_pure_time);
        resultDB.AddResult(testName+"-N_PCIe", toString(N), "GFLOPS",
                flops / user_wait_time);
        resultDB.AddResult(testName+"-N_Parity", toString(N), "N",
                transfer_time / gemm_pure_time);
    }

    // Run NT
    for (int i = 0; i < passes; i++) {
        Event evDownload1("Download A");
        Event evUpload("Upload");
        Event evNT("sgemmNT");

        err = clEnqueueWriteBuffer(queue, Agpu, CL_TRUE, 0, m*n*sizeof(T),
                A, 0, NULL, &evDownload1.CLEvent());
        err = clEnqueueWriteBuffer(queue, Bgpu, CL_TRUE, 0, m*n*sizeof(T),
                B, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                C, 0, NULL, NULL);
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        //Launch Kernels
        err = clEnqueueNDRangeKernel(queue, sgemmNT, 2, NULL, globalWorkSize,
                                     localWorkSize, 0, NULL, &evNT.CLEvent());
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        err = clEnqueueReadBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                C, 0, NULL, &evUpload.CLEvent());
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        evNT.FillTimingInfo();
        evDownload1.FillTimingInfo();
        evUpload.FillTimingInfo();

        double user_wait_time = 0.0;
        double gemm_pure_time = 0.0;

        gemm_pure_time = evNT.SubmitEndRuntime();
        user_wait_time = evUpload.EndTime() - evDownload1.QueuedTime();
        double transfer_time = user_wait_time - gemm_pure_time;
        double flops = 2.0*(double)N*N*N;
        resultDB.AddResult(testName+"-T", toString(N), "GFLOPS",
                flops / gemm_pure_time);
        resultDB.AddResult(testName+"-T_PCIe", toString(N), "GFLOPS",
                flops / user_wait_time);
        resultDB.AddResult(testName+"-T_Parity", toString(N), "N",
                transfer_time / gemm_pure_time);
    }

    if (true) // pinned
    {
        err = clReleaseMemObject(Aobj);
        CL_CHECK_ERROR(err);
        err = clReleaseMemObject(Bobj);
        CL_CHECK_ERROR(err);
        err = clReleaseMemObject(Cobj);
        CL_CHECK_ERROR(err);
    }
    else
    {
	free(A);
	free(B);
	free(C);
    }

    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(sgemmNN);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(sgemmNT);
    CL_CHECK_ERROR(err);

    err = clReleaseMemObject(Agpu);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(Bgpu);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(Cgpu);
    CL_CHECK_ERROR(err);

}
