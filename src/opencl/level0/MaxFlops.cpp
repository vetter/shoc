
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "support.h"
#include "Event.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "ProgressBar.h"

using namespace std;

// OpenCL kernels are auto-generated based on the fields of this structure.
// Enables easy modification of the main benchmark parameters.
struct _benchmark_type {
   const char* name;             // name of the kernel
   const char* indexVar;         // name of the private scalar used as
                                 // an accumulator
   const char* indexVarInit;     // initialization formula for the index
                                 // variable
   const char* opFormula;        // arithmetic formula for the accumulator
   int numStreams;               // number of parallel streams
   int numUnrolls;               // number of times the loop was unrolled
   int numRepeats;               // number of loop iterations (>= 1)
   int flopCount;                // number of floating point operations in one
                                 // formula
   int halfBufSizeMin;           // specify the buffer sizes for which to
                                 // perform the test
   int halfBufSizeMax;           // we specify the minimum, the maximum and
                                 // the geometric stride
   int halfBufSizeStride;        // values are in thousands of elements
} aTests[] = {
  {"Add1", "s", "data[gid]", "10.f-$", 1, 240, 20, 1, 1024, 1024, 4},
  {"Add2", "s", "data[gid]", "10.f-$", 2, 120, 20, 1, 1024, 1024, 4},
  {"Add4", "s", "data[gid]", "10.f-$", 4, 60, 20, 1, 1024, 1024, 4},
  {"Add8", "s", "data[gid]", "10.f-$", 8, 30, 20, 1, 1024, 1024, 4},
  {"Add16", "s", "data[gid]", "10.f-$", 16, 20, 20, 1, 1024, 1024, 4},
  {"Mul1", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 1, 200, 20, 2, 1024, 1024, 4},
  {"Mul2", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 2, 100, 20, 2, 1024, 1024, 4},
  {"Mul4", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 4, 50, 20, 2, 1024, 1024, 4},
  {"Mul8", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 8, 25, 20, 2, 1024, 1024, 4},
  {"Mul16", "s", "data[gid]-data[gid]+0.999f", "$*$*1.01f", 16, 15, 20, 2, 1024, 1024, 4},
  {"MAdd1", "s", "data[gid]", "10.0f-$*0.9899f", 1, 240, 20, 2, 1024, 1024, 4},
  {"MAdd2", "s", "data[gid]", "10.0f-$*0.9899f", 2, 120, 20, 2, 1024, 1024, 4},
  {"MAdd4", "s", "data[gid]", "10.0f-$*0.9899f", 4, 60, 20, 2, 1024, 1024, 4},
  {"MAdd8", "s", "data[gid]", "10.0f-$*0.9899f", 8, 30, 20, 2, 1024, 1024, 4},
  {"MAdd16", "s", "data[gid]", "10.0f-$*0.9899f", 16, 20, 20, 2, 1024, 1024, 4},
  {"MulMAdd1", "s", "data[gid]", "(3.75f-0.355f*$)*$", 1, 160, 20, 3, 1024, 1024, 4},
  {"MulMAdd2", "s", "data[gid]", "(3.75f-0.355f*$)*$", 2, 80, 20, 3, 1024, 1024, 4},
  {"MulMAdd4", "s", "data[gid]", "(3.75f-0.355f*$)*$", 4, 40, 20, 3, 1024, 1024, 4},
  {"MulMAdd8", "s", "data[gid]", "(3.75f-0.355f*$)*$", 8, 20, 20, 3, 1024, 1024, 4},
  {0, 0, 0, 0, 0, 0, 0, 0, 0}
};


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
// Programmer: Gabriel Marin
// Creation: July 08, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    ; // this benchmark has no specific options
}

// OpenCL compiler options -- default is to enable
// all optimizations
static const char* opts = "-cl-mad-enable -cl-no-signed-zeros "
                          "-cl-unsafe-math-optimizations -cl-finite-math-only";

// Forward Declarations
// generate simple precision and double precision versions of the benchmarks
template <class T> void
RunTest(cl_device_id id, cl_context ctx, cl_command_queue queue, ResultDatabase &resultDB,
        int npasses, int verbose, int quiet, float repeatF, size_t localSize, ProgressBar &pb,
        const char* typeName, const char* precision, const char* pragmaText);
// Generate OpenCL kernel code based on benchmark type struct
void generateKernel (ostringstream &oss, struct _benchmark_type &test, const char* tName, const char* header);
// Generate kernel code for a highly unrolled MADD or MADDMUL bm
string generateUKernel(int useMADDMUL, bool doublePrecision, int nRepeats, int nUnrolls,
             const char* tName, const char* header);
// Get flops/work item for custom kernels
double getUFlopCount(int useMADDMUL, bool doublePrecision, int nRepeats, int nUnrolls);

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes a series of arithmetic benchmarks for OpenCL devices.
//   OpenCL kernels are auto-generated based on the values in the
//   _benchmark_type structures.
//   The benchmark tests throughput for add, multiply, multiply-add and
//   multiply+multiply-add series of operations, for 1, 2, 4 and 8
//   independent streams..
//
// Arguments:
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser (contains input parameters)
//
// Returns:  nothing
//
// Programmer: Gabriel Marin
// Creation: June 26, 2009
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
    int npasses  = op.getOptionInt("passes");
    bool verbose = op.getOptionBool("verbose");
    bool quiet = op.getOptionBool("quiet");


    int err;
    cl_mem mem1;
    float *hostMem, *hostMem2;
    size_t maxGroupSize = 1;
    size_t localWorkSize = 1;

    // Seed the random number generator
    srand48(8650341L);

    // To prevent this benchmark from taking too long to run, we
    // calibrate how many repetitions of each test to execute. To do this we
    // run one pass through a multiply-add benchmark and then adjust
    // the repeat factor based on runtime. Use MulMAdd4 for this.
    int aIdx = 0;
    float repeatF = 1.0f;
    // Find the index of the MAdd4 benchmark
    while ((aTests!=0) && (aTests[aIdx].name!=0) &&
        strcmp(aTests[aIdx].name,"MAdd4"))
    {
       aIdx += 1;
    }
    if (aTests && aTests[aIdx].name)  // we found a benchmark with that name
    {
       struct _benchmark_type temp = aTests[aIdx];
       // Limit to one repetition
       temp.numRepeats = 10;

       // Kernel will be generated into this stream
       ostringstream oss;
       generateKernel (oss, temp, "float", "");
       std::string kernelCode(oss.str());

       // Allocate host memory
       int halfNumFloatsMax = temp.halfBufSizeMax*1024/4;
       int numFloatsMax = 2*halfNumFloatsMax;
       hostMem = new float[numFloatsMax];
       hostMem2 = new float[numFloatsMax];

       // Allocate device memory
       mem1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 sizeof(float)*numFloatsMax, NULL, &err);
       CL_CHECK_ERROR(err);

       err = clEnqueueWriteBuffer(queue, mem1, true, 0,
                               numFloatsMax*sizeof(float), hostMem,
                               0, NULL, NULL);
       CL_CHECK_ERROR(err);

       // Create the kernel program
       const char* progSource[] = {kernelCode.c_str()};
       cl_program prog = clCreateProgramWithSource(ctx, 1, progSource,
                                                   NULL, &err);
       CL_CHECK_ERROR(err);

       // Compile the kernel
       err = clBuildProgram(prog, 0, NULL, opts, NULL, NULL);
       // Compile the kernel
       CL_CHECK_ERROR(err);

       // Extract out madd kernel
       cl_kernel kernel_madd = clCreateKernel(prog, temp.name, &err);
       CL_CHECK_ERROR(err);

       // Set kernel arguments
       err = clSetKernelArg (kernel_madd, 0, sizeof(cl_mem), (void*)&mem1);
       CL_CHECK_ERROR (err);
       err = clSetKernelArg (kernel_madd, 1, sizeof(cl_int),
                             (void*)&temp.numRepeats);
       CL_CHECK_ERROR (err);

       // Determine the maximum work group size for this kernel
       maxGroupSize = getMaxWorkGroupSize(id);
       // use min(maxWorkGroupSize, 256)
       localWorkSize = maxGroupSize<128?maxGroupSize:128;

       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloatsMax; ++j)
       {
           hostMem[j] = hostMem[numFloatsMax-j-1] = (float)(drand48()*5.0);
       }
       // Set global work size
       size_t globalWorkSize = numFloatsMax;

       Event evCopyMem("CopyMem");
       err = clEnqueueWriteBuffer (queue, mem1, true, 0,
                                   numFloatsMax*sizeof(float), hostMem,
                                   0, NULL, &evCopyMem.CLEvent());
       CL_CHECK_ERROR (err);
       // Wait for transfer to finish
       err = clWaitForEvents (1, &evCopyMem.CLEvent());
       CL_CHECK_ERROR (err);

       Event evKernel(temp.name);
       err = clEnqueueNDRangeKernel (queue, kernel_madd, 1, NULL,
                                 &globalWorkSize, &localWorkSize,
                                 0, NULL, &evKernel.CLEvent());
       CL_CHECK_ERROR (err);
       // Wait for kernel to finish
       err = clWaitForEvents (1, &evKernel.CLEvent());
       CL_CHECK_ERROR (err);

       evKernel.FillTimingInfo();
       // Calculate repeat factor based on kernel runtime
       double tt = double(evKernel.SubmitEndRuntime());
       repeatF = 1.1e07 / tt;
       cout << "Adjust repeat factor = " << repeatF << endl;

       // Clean up
       err = clReleaseKernel (kernel_madd);
       CL_CHECK_ERROR(err);
       err = clReleaseProgram (prog);
       CL_CHECK_ERROR(err);
       err = clReleaseMemObject(mem1);
       CL_CHECK_ERROR(err);

       delete[] hostMem;
       delete[] hostMem2;
    }

    // Compute total number of kernel runs
    int totalRuns = 0;
    aIdx = 0;
    while ((aTests!=0) && (aTests[aIdx].name!=0))
    {
        for (int halfNumFloats=aTests[aIdx].halfBufSizeMin*1024 ;
             halfNumFloats<=aTests[aIdx].halfBufSizeMax*1024 ;
             halfNumFloats*=aTests[aIdx].halfBufSizeStride)
        {
            totalRuns += npasses;
        }
        aIdx += 1;
    }
    // Account for custom kernels
    totalRuns += 2 * npasses;

    // check for double precision support
    int hasDoubleFp = 0;
    string doublePragma = "";
    if (checkExtension(id, "cl_khr_fp64")) {
        hasDoubleFp = 1;
        doublePragma = "#pragma OPENCL EXTENSION cl_khr_fp64: enable";
    } else
    if (checkExtension(id, "cl_amd_fp64")) {
        hasDoubleFp = 1;
        doublePragma = "#pragma OPENCL EXTENSION cl_amd_fp64: enable";
    }

    // Double the number of passes if double precision support found
    if (hasDoubleFp) {
        cout << "DP Supported" << endl;
        totalRuns <<= 1;
    } else
        cout << "DP Not Supported" << endl;

    ProgressBar pb(totalRuns);
    if (!verbose && !quiet)
        pb.Show(stdout);

    RunTest<float> (id, ctx, queue, resultDB, npasses, verbose, quiet,
             repeatF, localWorkSize, pb, "float", "-SP", "");

    if (hasDoubleFp)
        RunTest<double> (id, ctx, queue, resultDB, npasses, verbose, quiet,
             repeatF, localWorkSize, pb, "double", "-DP", doublePragma.c_str());
    else
    {
        aIdx = 0;
        const char atts[] = "DP_Not_Supported";
        while ((aTests!=0) && (aTests[aIdx].name!=0))
        {
            for (int pas=0 ; pas<npasses ; ++pas)
            {
                resultDB.AddResult(string(aTests[aIdx].name)+"-DP" , atts, "GFLOPS", FLT_MAX);
            }
            aIdx += 1;
        }
        for (int pas=0 ; pas<npasses ; ++pas)
        {
            resultDB.AddResult("MulMAddU-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("MAddU-DP", atts, "GFLOPS", FLT_MAX);
        }
    }
    if (!verbose)
        fprintf (stdout, "\n\n");
}

template <class T> void
RunTest(cl_device_id id,
        cl_context ctx,
        cl_command_queue queue,
        ResultDatabase &resultDB,
        int npasses,
        int verbose,
        int quiet,
        float repeatF,
        size_t localWorkSize,
        ProgressBar &pb,
        const char* typeName,
        const char* precision,
        const char* pragmaText)
{
    int err;
    cl_mem mem1;
    char sizeStr[128];
    T *hostMem, *hostMem2;

    int aIdx = 0;
    while ((aTests!=0) && (aTests[aIdx].name!=0))
    {
       ostringstream oss;
       struct _benchmark_type temp = aTests[aIdx];

       // Calculate adjusted repeat factor
       int tentativeRepeats = (int)round(repeatF*temp.numRepeats);
       if (tentativeRepeats < 2) {
          tentativeRepeats = 2;
          double realRepeatF = ((double)tentativeRepeats)
            / temp.numRepeats;
          if (realRepeatF>8.0*repeatF)  // do not cut the number of unrolls
                                        // by more than a factor of 8
             realRepeatF = 8.0*repeatF;
          temp.numUnrolls =
                  (int)round(repeatF*temp.numUnrolls/realRepeatF);
       }
       temp.numRepeats = tentativeRepeats;

       // Generate kernel source code
       generateKernel(oss, temp, typeName, pragmaText);
       std::string kernelCode(oss.str());

       // If in verbose mode, print the kernel
       if (verbose)
       {
           cout << "Code for kernel " << temp.name
                << ":\n" + kernelCode << endl;
       }

       // Alloc host memory
       int halfNumFloatsMax = temp.halfBufSizeMax*1024;
       int numFloatsMax = 2*halfNumFloatsMax;
       hostMem = new T[numFloatsMax];
       hostMem2 = new T[numFloatsMax];

       // Allocate device memory
       mem1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 sizeof(T)*numFloatsMax, NULL, &err);
       CL_CHECK_ERROR(err);

       // Issue a copy to force device allocation
       err = clEnqueueWriteBuffer(queue, mem1, true, 0,
                               numFloatsMax*sizeof(T), hostMem,
                               0, NULL, NULL);
       CL_CHECK_ERROR(err);

       // Create kernel program object
       const char* progSource[] = {kernelCode.c_str()};
       cl_program prog = clCreateProgramWithSource(ctx, 1, progSource,
           NULL, &err);
       CL_CHECK_ERROR(err);

       // Compile the program
       err = clBuildProgram(prog, 1, &id, opts, NULL, NULL);
       CL_CHECK_ERROR(err);

       if (err != 0)
       {
           char log[5000];
           size_t retsize = 0;
           err =  clGetProgramBuildInfo(prog, id, CL_PROGRAM_BUILD_LOG,
                    5000*sizeof(char), log, &retsize);
           CL_CHECK_ERROR(err);

           cout << "Build error." << endl;
           cout << "Retsize: " << retsize << endl;
           cout << "Log: " << log << endl;
           return;
       }

       // Check if we have to dump the PTX (NVIDIA only)
       // Disabled by default
       // Set environment variable DUMP_PTX to enable
       char* dumpPtx = getenv("DUMP_PTX");
       if (dumpPtx && !strcmp(dumpPtx, "1")) {  // must dump the PTX
          dumpPTXCode(ctx, prog, temp.name);
       }

       // Extract out kernel
       cl_kernel kernel_madd = clCreateKernel(prog, temp.name, &err);
       CL_CHECK_ERROR(err);

       err = clSetKernelArg (kernel_madd, 0, sizeof(cl_mem), (void*)&mem1);
       CL_CHECK_ERROR (err);
       err = clSetKernelArg (kernel_madd, 1, sizeof(cl_int),
           (void*)&temp.numRepeats);
       CL_CHECK_ERROR (err);

       if (verbose)
       {
          cout << "Running kernel " << temp.name << endl;
       }

       for (int halfNumFloats=temp.halfBufSizeMin*1024 ;
            halfNumFloats<=temp.halfBufSizeMax*1024 ;
            halfNumFloats*=temp.halfBufSizeStride)
       {
          // Set up input memory, first half = second half
          int numFloats = 2*halfNumFloats;
          for (int j=0; j<halfNumFloats; ++j)
          {
             hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*5.0);
          }

          size_t globalWorkSize = numFloats;

          for (int pas=0 ; pas<npasses ; ++pas)
          {
             err = clEnqueueWriteBuffer (queue, mem1, true, 0,
                                   numFloats*sizeof(T), hostMem,
                                   0, NULL, NULL);
             CL_CHECK_ERROR(err);

             Event evKernel(temp.name);

             err = clEnqueueNDRangeKernel(queue, kernel_madd, 1, NULL,
                                 &globalWorkSize, &localWorkSize,
                                 0, NULL, &evKernel.CLEvent());
             CL_CHECK_ERROR(err);

             err = clWaitForEvents(1, &evKernel.CLEvent());
             CL_CHECK_ERROR(err);

             evKernel.FillTimingInfo();
             double flopCount = (double)numFloats *
                                temp.flopCount *
                                temp.numRepeats *
                                temp.numUnrolls *
                                temp.numStreams;

             double gflop = flopCount / (double)(evKernel.SubmitEndRuntime());

             sprintf (sizeStr, "Size:%07d", numFloats);
             resultDB.AddResult(string(temp.name)+precision, sizeStr, "GFLOPS", gflop);

             // Zero out the test host memory
             for (int j=0 ; j<numFloats ; ++j)
             {
                 hostMem2[j] = 0.0;
             }

             // Read the result device memory back to the host
             err = clEnqueueReadBuffer(queue, mem1, true, 0,
                                 numFloats*sizeof(T), hostMem2,
                                 0, NULL, NULL);
             CL_CHECK_ERROR(err);

             // Check the result -- At a minimum the first half of memory
             // should match the second half exactly
             for (int j=0 ; j<halfNumFloats ; ++j)
             {
                if (hostMem2[j] != hostMem2[numFloats-j-1])
                {
                   cout << "Error; hostMem2[" << j << "]=" << hostMem2[j]
                        << " is different from its twin element hostMem2["
                        << (numFloats-j-1) << "]=" << hostMem2[numFloats-j-1]
                        <<"; stopping check\n";
                   break;
                }
             }

             // update progress bar
             pb.addItersDone();
             if (!verbose && !quiet)
                 pb.Show(stdout);
          }
       }

       err = clReleaseKernel (kernel_madd);
       CL_CHECK_ERROR(err);
       err = clReleaseProgram (prog);
       CL_CHECK_ERROR(err);
       err = clReleaseMemObject(mem1);
       CL_CHECK_ERROR(err);

       aIdx += 1;

       delete[] hostMem;
       delete[] hostMem2;
    }

    // Now, test hand-tuned custom kernels

    // 2D - width and height of input
    const int w = 2048, h = 2048;
    const int bytes = w * h * sizeof(T);

    // Allocate some device memory
    mem1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);

    // Get a couple non-zero random numbers
    float val1 = 0, val2 = 0;
    while (val1==0 || val2==0)
    {
        val1 = drand48();
        val2 = drand48();
    }
    // For each custom kernel
    for (int kCounter = 0; kCounter < 2; kCounter++)
    {
        // Calculate adjusted repeat factor
        int tentativeRepeats = (int)round(repeatF*5);
        int nUnrolls = 100;
        if (tentativeRepeats < 2) {
           tentativeRepeats = 2;
           double realRepeatF = ((double)tentativeRepeats) / 5;
           if (realRepeatF>8.0*repeatF)  // do not cut the number of unrolls
                                         // by more than a factor of 8
              realRepeatF = 8.0*repeatF;
           nUnrolls = (int)round(repeatF*100/realRepeatF);
        }

        // Double precision not currently supported
        string kSource = generateUKernel(kCounter, false, tentativeRepeats, nUnrolls,
                       typeName, pragmaText);

        const char* progSource[] = {kSource.c_str()};
        cl_program prog = clCreateProgramWithSource(ctx,
                                 1, progSource, NULL, &err);
        CL_CHECK_ERROR(err);

        // Compile kernel
        err = clBuildProgram(prog, 1, &id, opts, NULL, NULL);
        CL_CHECK_ERROR(err);

        // Extract out kernel
        cl_kernel kernel_madd = clCreateKernel(prog, "peak", &err);

        // Calculate kernel launch parameters
        //size_t localWorkSize = maxGroupSize<128?maxGroupSize:128;
        size_t globalWorkSize = w * h;

        // Set the arguments
        err = clSetKernelArg(kernel_madd, 0, sizeof(cl_mem), (void*)&mem1);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(kernel_madd, 1, sizeof(T), (void*)&val1);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(kernel_madd, 2, sizeof(T), (void*)&val2);
        CL_CHECK_ERROR(err);

        for (int passCounter=0; passCounter < npasses; passCounter++)
        {
            // Event object for timing
            Event evKernel_madd("madd");

            err = clEnqueueNDRangeKernel(queue, kernel_madd, 1, NULL,
                      &globalWorkSize, &localWorkSize,
                      0, NULL, &evKernel_madd.CLEvent());
            CL_CHECK_ERROR(err);

            // Wait for the kernel to finish
            err = clWaitForEvents(1, &evKernel_madd.CLEvent());
            CL_CHECK_ERROR(err);
            evKernel_madd.FillTimingInfo();

            // Calculate result and add to DB
            char atts[1024];
            double nflopsPerItem = getUFlopCount(kCounter, false, tentativeRepeats, nUnrolls);
            sprintf(atts, "Size:%d", w*h);
            double gflops = (double) (nflopsPerItem*w*h) /
                            (double) evKernel_madd.SubmitEndRuntime();

            if (kCounter) {
                resultDB.AddResult(string("MulMAddU")+precision, atts, "GFLOPS", gflops);
            } else {
                resultDB.AddResult(string("MAddU")+precision, atts, "GFLOPS", gflops);
            }
            // update progress bar
            pb.addItersDone();
            if (!verbose && !quiet)
            {
                pb.Show(stdout);
            }
        }
        err = clReleaseKernel(kernel_madd);
        CL_CHECK_ERROR(err);
        err = clReleaseProgram(prog);
        CL_CHECK_ERROR(err);
    }
    err = clReleaseMemObject(mem1);
    CL_CHECK_ERROR(err);
}




// ****************************************************************************
// Function: generateKernel
//
// Purpose:
//   Generate an OpenCL kernel based on the content of the _benchmark_type
//   structure.
//
// Arguments:
//   oss: output string stream for writing the generated kernel
//   test: structure containing the benchmark parameters
//
// Returns:  nothing
//
// Programmer: Gabriel Marin
// Creation: June 26, 2009
//
// Modifications:
//
// ****************************************************************************
void
generateKernel (ostringstream &oss, struct _benchmark_type &test, const char* tName, const char* header)
{
    string kName = test.name;
    oss << header << endl;
    oss << string("__kernel void ") << kName << "(__global " << tName << " *data, int nIters) {\n"
        << "  int gid = get_global_id(0), globalSize = get_global_size(0);\n";
    // use vector types to store the index variables when the number of streams is > 1
    // OpenCL has vectors of length 1 (scalar), 2, 4, 8, 16. Use the largest vector possible.
    // keep track of how many vectors of each size we are going to use.
    int numVecs[5] = {0, 0, 0, 0, 0};
    int startIdx[5] = {0, 0, 0, 0, 0};
    int i, nStreams = test.numStreams, nVectors = 0;
    oss << "  " << tName << " " << test.indexVar << " = " << test.indexVarInit << ";\n";
    float iniVal = 0.f;
    for (i=4 ; i>=0 ; --i)
    {
       numVecs[i] = nStreams / (1<<i);
       nStreams -= (numVecs[i]*(1<<i));
       if (i==4) startIdx[i] = 0;
       else startIdx[i] = startIdx[i+1] + numVecs[i+1];
       nVectors += numVecs[i];

       for (int vv=startIdx[i] ; vv<startIdx[i]+numVecs[i] ; ++vv)
       {
          oss << "  " << tName;
          if (i>0) oss << (1<<i);
          oss << " " << test.indexVar << vv << " = "
              << test.indexVar << " + ";
          if (i>0) oss << "(" << tName << (1<<i) << ")(";
          oss << iniVal;
          iniVal += 0.1;
          for (int ss=1 ; ss<(1<<i) ; ++ss) {
             oss << "," << iniVal;
             iniVal += 0.1;
          }
          if (i>0) oss << ")";
          oss << ";\n";
       }
    }
    if (test.numRepeats > 1)
        oss << "  for (int j=0 ; j<nIters ; ++j) {\n";

    // write the body of the loop
    char buf[32];
    for (int uu=0 ; uu<test.numUnrolls ; ++uu) {
        for (int ss=0 ; ss<nVectors ; ++ss)
        {
            string opCode = string(test.opFormula);
            int pos = -1;
            sprintf (buf, "%s%d", test.indexVar, ss);
            string lVar = string(buf);
            while ((pos=opCode.find("$"))!=(-1))
                opCode.replace(pos,1,lVar);
            oss << " " << lVar << "=" << opCode << ";";
        }
        oss << "\n";
    }

    if (test.numRepeats > 1)
        oss << "  }\n";

    // now sum up all the vectors; I do not actually care about the numerical result,
    // only to mark the values as live for the compiler. So sum up the vectors even if
    // some of them will end up being expanded. Then some the terms of the first vector;
    for (i=4 ; i>=0 ; --i)
       if (numVecs[i]>1)  // sum all vectors of same length
       {
          oss << "   " << test.indexVar << startIdx[i] << " = " << test.indexVar << startIdx[i];
          for (int ss=startIdx[i]+1 ; ss<startIdx[i]+numVecs[i] ; ++ss)
             oss << "+" << test.indexVar << ss;
          oss << ";\n";
       }
    oss << "   data[gid] = ";
    // find the size of the largest vector use;
    bool first = true;
    for (i=4 ; i>=0 ; --i)
       if (numVecs[i]>0)  // this is it
       {
          for (int ss=0 ; ss<(1<<i) ; ++ss)
          {
             if (! first) {
                oss << "+";
             } else
                first = false;
             oss << test.indexVar << startIdx[i];
             if (i>0)
                oss << ".s" << hex << ss << dec;
          }
       }
    oss << ";\n}";
}

// ****************************************************************************
// Function: getUFlopCount
//
// Purpose:
//   Calculate the number of FLOPS per work item for custom unrolled kernels.
//   The parameters of this method select the kernel of interest.
//
// Arguments:
//   useMADDMUL: 0 for MADD only, 1 for MADD+MUL
//   doublePrecision: true if double precision, false for single
//
// Returns:  number of flops per work item
//
// Programmer: Gabriel Marin
// Creation: June 26, 2009
//
// Modifications:
//
// ****************************************************************************

double getUFlopCount(int useMADDMUL, bool doublePrecision, int nRepeats, int nUnrolls) {
    if (useMADDMUL) {
        return ((3.0*8.0)*nUnrolls*nRepeats) + 61.0;
    } else {
        return ((2.0*32.0)*nUnrolls*nRepeats) + 61.0;
    }
}
// ****************************************************************************
// Function: generateUKernel
//
// Purpose:
//   Generate a custom unrolled OpenCL kernel to measure max FLOPS
//
//
// Arguments:
//   useMADDMUL: 0 for MADD only, 1 for MADD+MUL
//   doublePrecision: true if double precision, false for single
//
// Returns: the kernel code
//
// Programmer: Gabriel Marin
// Creation: June 26, 2009
//
// Modifications:
//
// ****************************************************************************

string
generateUKernel(int useMADDMUL, bool doublePrecision, int nRepeats, int nUnrolls,
             const char* tName, const char* header)
{
    string ops;
    char bufrep[16];

    sprintf(bufrep, "%d", nRepeats);

    if (useMADDMUL) {
        ops = "s0 = s4*s4 + s4; \
               s6 = s0*s5;      \
               s1 = s5*s5 + s5; \
               s7 = s1*s6;      \
               s2 = s6*s6 + s6; \
               s0 = s2*s7;      \
               s3 = s7*s7 + s7; \
               s1 = s3*s0;      \
               s4 = s0*s0 + s0; \
               s2 = s4*s1;      \
               s5 = s1*s1 + s1; \
               s3 = s5*s2;      \
               s6 = s2*s2 + s2; \
               s4 = s6*s3;      \
               s7 = s3*s3 + s3; \
               s5 = s7*s4;";

    } else {
        ops = "s0 = s6*s5 + s28;                       \
               s1 = s7*s6 + s29;                       \
               s2 = s8*s7 + s30;                       \
               s3 = s9*s8 + s31;                       \
               s4 = s10*s9 + s0;                       \
               s5 = s11*s10 + s1;                      \
               s6 = s12*s11 + s2;                      \
               s7 = s13*s12 + s3;                      \
               s8 = s14*s13 + s4;                      \
               s9 = s15*s14 + s5;                      \
               s10 = s16*s15 + s6;                     \
               s11 = s17*s16 + s7;                     \
               s12 = s18*s17 + s8;                     \
               s13 = s19*s18 + s9;                     \
               s14 = s20*s19 + s10;                    \
               s15 = s21*s20 + s11;                    \
               s16 = s22*s21 + s12;                    \
               s17 = s23*s22 + s13;                    \
               s18 = s24*s23 + s14;                    \
               s19 = s25*s24 + s15;                    \
               s20 = s26*s25 + s16;                    \
               s21 = s27*s26 + s17;                    \
               s22 = s28*s27 + s18;                    \
               s23 = s29*s28 + s19;                    \
               s24 = s30*s29 + s20;                    \
               s25 = s31*s30 + s21;                    \
               s26 = s0*s31 + s22;                     \
               s27 = s1*s0 + s23;                      \
               s28 = s2*s1 + s24;                      \
               s29 = s3*s2 + s25;                      \
               s30 = s4*s3 + s26;                      \
               s31 = s5*s4 + s27;";
    }
    string op100 = " ";
    for (int k=0; k<nUnrolls;k++) op100 += ops;
    string kSource = string(header) + "\n"
    "__kernel void peak(__global " + tName +" *target, " + tName + " val1, " + tName + " val2) "
    "{                                                                      "
    " int index = get_global_id(0);                                         "
    " " + tName + " v0=val1,     v1=val2,     v2=v0+v1,    v3=v0+v2;                "
    " " + tName + " v4=v0+v3,    v5=v0+v4,    v6=v0+v5,    v7=v0+v6;                "
    " " + tName + " v8=v0+v7,    v9=v0+v8,    v10=v0+v9,   v11=v0+v10;              "
    " " + tName + " v12=v0+v11,  v13=v0+v12,  v14=v0+v13,  v15=v0+v14;              "
    " " + tName + " v16=v0+v15,  v17=v16+v0,  v18=v16+v1,  v19=v16+v2;              "
    " " + tName + " v20=v16+v3,  v21=v16+v4,  v22=v16+v5,  v23=v16+v6;              "
    " " + tName + " v24=v16+v7,  v25=v16+v8,  v26=v16+v9,  v27=v16+v10;             "
    " " + tName + " v28=v16+v11, v29=v16+v12, v30=v16+v13, v31=v16+v14;             "
    " " + tName + " s0=v0,   s1=v1,   s2=v2,   s3=v3;                               "
    " " + tName + " s4=v4,   s5=v5,   s6=v6,   s7=v7;                               "
    " " + tName + " s8=v8,   s9=v9,   s10=v10, s11=v11;                             "
    " " + tName + " s12=v12, s13=v13, s14=v14, s15=v15;                             "
    " " + tName + " s16=v16, s17=v17, s18=v18, s19=v19;                             "
    " " + tName + " s20=v20, s21=v21, s22=v22, s23=v23;                             "
    " " + tName + " s24=v24, s25=v25, s26=v26, s27=v27;                             "
    " " + tName + " s28=v28, s29=v29, s30=v30, s31=v31;                             "
    " int ctr;                                                              "
    " for (ctr=0; ctr<" + string(bufrep) + "; ctr++) {                                           "
    + op100 +
    " }                                                                     "
    " " + tName + " result = (s0+s1+s2+s3+s4+s5+s6+s7+                              "
    "                 s8+s9+s10+s11+s12+s13+s14+s15 +                       "
    "                s16+s17+s18+s19+s20+s21+s22+s23+                       "
    "                s24+s25+s26+s27+s28+s29+s30+s31);                      "
    " target[index] = result;                                               "
    " } ";

    return kSource;
}

