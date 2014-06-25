#include <iostream>
#include "support.h"
#include "Event.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

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
   ; // This benchmark has no specific options of its own
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the submit start delay benchmark.  This benchmark
//   is designed to test the OpenCL's work queue implementation.
//   It tries to examine the delay from when something is submitted
//   to the OpenCL work queue until it actually starts executing.
//   It measures this average delay for a single kernel, as well
//   as alternating between 2 and 4 kernels.
//
// Arguments:
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser (contains input parameters)
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(cl_device_id id,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    bool verbose = op.getOptionBool("verbose");

    // Number of iterations to use in measuring delay
    int reps = 32;
    // Repeat the test 3 times
    int passes = 3;

    const bool waitForEvents = true;
    int err = 0;

    // Create a couple test kernels
    if (verbose) cout << ">> building the kernel\n";
    const char *plusOneCLSource[] = {
        "__kernel   void one(__global int* a) {int aa;}",
        "__kernel   void two(__global int* b) {int bb;}",
        "__kernel void three(__global int* c) {int cc;}",
        "__kernel  void four(__global int* d) {int dd;}"
    };
    cl_program prog = clCreateProgramWithSource(ctx,
                                                4, plusOneCLSource, NULL,
                                                &err);
    CL_CHECK_ERROR(err);

    // Compile the program
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    CL_CHECK_ERROR(err);

    // If there is a build error, print the output and return
    if (err != CL_SUCCESS)
    {
        char log[5000];
        size_t retsize = 0;
        err = clGetProgramBuildInfo(prog, id, CL_PROGRAM_BUILD_LOG, 50000
                * sizeof(char), log, &retsize);
        CL_CHECK_ERROR(err);
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        return;
    }

    // Extract out kernel
    cl_kernel kernel1 = clCreateKernel(prog, "one", &err);
    CL_CHECK_ERROR(err);

    // Extract out "plus-two" kernel
    cl_kernel kernel2 = clCreateKernel(prog, "two", &err);
    CL_CHECK_ERROR(err);

    // Extract out "plus-three" kernel
    cl_kernel kernel3 = clCreateKernel(prog, "three", &err);
    CL_CHECK_ERROR(err);

    // Extract out "plus-four" kernel
    cl_kernel kernel4 = clCreateKernel(prog, "four", &err);
    CL_CHECK_ERROR(err);

    cl_mem zero = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
            sizeof(cl_int), NULL, &err);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(kernel1, 0, sizeof(cl_mem),
            (void*) &zero);
    err = clSetKernelArg(kernel2, 0, sizeof(cl_mem),
            (void*) &zero);
    err = clSetKernelArg(kernel3, 0, sizeof(cl_mem),
            (void*) &zero);
    err = clSetKernelArg(kernel4, 0, sizeof(cl_mem),
            (void*) &zero);

    size_t maxGroupSize = getMaxWorkGroupSize(ctx, kernel1);
    size_t localWorkSize = (maxGroupSize >= 256 ? 256 : maxGroupSize);
    size_t globalWorkSize = localWorkSize * 256;

    // Test single kernel
    for (int j = 0; j < passes; j++)
    {
       double total = 0.0;
       for (int i = 0; i < reps; i++)
       {
          // Declare event objects for the kernels
          Event evKernel1("Run Kernel1");
          Event evKernel2("Run Kernel2");
          Event evKernel3("Run Kernel3");
          Event evKernel4("Run Kernel4");

          err = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel1.CLEvent());
          err = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel2.CLEvent());
          err = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel3.CLEvent());
          err = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel4.CLEvent());
          CL_CHECK_ERROR(err);

          // Wait for the kernels to finish
          if (waitForEvents)
          {
             err = clWaitForEvents(1, &evKernel4.CLEvent());
             CL_CHECK_ERROR(err);
          }

          evKernel1.FillTimingInfo();
          evKernel2.FillTimingInfo();
          evKernel3.FillTimingInfo();
          evKernel4.FillTimingInfo();

          total += evKernel1.SubmitStartDelay() +
                   evKernel2.SubmitStartDelay() +
                   evKernel3.SubmitStartDelay() +
                   evKernel4.SubmitStartDelay();

       }
       resultDB.AddResult("SSDelay", "1 Kernel", "ms",
                          (total / ((double)reps * 4.0)) / 1.0e6 );
       total = 0.0;
    }

    // Perform the test alternating between two kernels
    for (int j = 0; j < passes; j++)
    {
       double total = 0.0;
       for (int i = 0; i < reps; i++)
       {
           // Declare event objects for the kernels
           Event evKernel1("Run Kernel1");
           Event evKernel2("Run Kernel2");
           Event evKernel3("Run Kernel3");
           Event evKernel4("Run Kernel4");

           err = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel1.CLEvent());
          CL_CHECK_ERROR(err);

          err = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel2.CLEvent());
          CL_CHECK_ERROR(err);

          err = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel3.CLEvent());
          CL_CHECK_ERROR(err);

          err = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel4.CLEvent());
          CL_CHECK_ERROR(err);


          // Wait for the kernel to finish
          if (waitForEvents)
          {
             err = clWaitForEvents(1, &evKernel4.CLEvent());
             CL_CHECK_ERROR(err);
          }

          evKernel1.FillTimingInfo();
          evKernel2.FillTimingInfo();
          evKernel3.FillTimingInfo();
          evKernel4.FillTimingInfo();
          total += evKernel1.SubmitStartDelay() +
                   evKernel2.SubmitStartDelay() +
                   evKernel3.SubmitStartDelay() +
                   evKernel4.SubmitStartDelay();

       }
       resultDB.AddResult("SSDelay", "2 Kernels", "ms",
                          (total / ((double)reps * 4.0)) / 1.0e6);
       total = 0.0;
    }

    // Perform the test alternating between four kernels
    for (int j = 0; j < passes; j++)
    {
       double total = 0.0;
       for (int i = 0; i < reps; i++)
       {
          // Declare event objects for the kernels
          Event evKernel1("Run Kernel1");
          Event evKernel2("Run Kernel2");
          Event evKernel3("Run Kernel3");
          Event evKernel4("Run Kernel4");

          err = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel1.CLEvent());
          CL_CHECK_ERROR(err);

          err = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel2.CLEvent());
          CL_CHECK_ERROR(err);

          err = clEnqueueNDRangeKernel(queue, kernel3, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel3.CLEvent());
          CL_CHECK_ERROR(err);

          err = clEnqueueNDRangeKernel(queue, kernel4, 1, NULL,
                                       &globalWorkSize, &localWorkSize,
                                       0, NULL, &evKernel4.CLEvent());
          CL_CHECK_ERROR(err);


          // Wait for the kernel to finish
          if (waitForEvents)
          {
             err = clWaitForEvents(1, &evKernel4.CLEvent());
             CL_CHECK_ERROR(err);
          }

          evKernel1.FillTimingInfo();
          evKernel2.FillTimingInfo();
          evKernel3.FillTimingInfo();
          evKernel4.FillTimingInfo();
          total += evKernel1.SubmitStartDelay() +
                   evKernel2.SubmitStartDelay() +
                   evKernel3.SubmitStartDelay() +
                   evKernel4.SubmitStartDelay();

       }
       resultDB.AddResult("SSDelay", "4 Kernels", "ms",
                          (total / ((double)reps * 4.0)) / 1.0e6);
       total = 0.0;
    }

    // Cleanup
    err = clReleaseKernel(kernel1);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(kernel2);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(kernel3);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(kernel4);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(zero);
    CL_CHECK_ERROR(err);
    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
}
