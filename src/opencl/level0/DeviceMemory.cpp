#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <signal.h>
#include "support.h"
#include "Event.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "ProgressBar.h"

using namespace std;

void TestImageMemory(cl_context ctx,
                     cl_command_queue queue,
                     ResultDatabase &resultDB,
                     OptionParser &op);

// define the data types for which the benchmarks are run
typedef enum {INT_TYPE, FLOAT_TYPE, NUM_ELEM_TYPES} ElemType;
const char* etypeNames[] = {"int", "float"};

// define the type of benchmark operations
typedef enum {OP_MEM_READ=1, OP_MEM_WRITE=2, NUM_MEMORY_OPERATIONS} OperationType;
const char* opNames[] = {"none", "read", "write"};

// OpenCL kernels are auto-generated based on the fields of this structure.
// Enables easy modification of the main benchmark parameters.
struct _benchmark_type {
   const char* name;             // name of the kernel
   const char* sourceType;       // type of the input array elements (relevant only for OP_MEM_READ)
   const char* destType;         // type of the output array elements (cannot be const)
   const char* indexVar;         // name of the private scalar used to index into the arrays
   const char* indexVarInit;     // initialization formula for the index variable
   bool  scaleInitFormula;       // scale initialization formula by size of memory per thread
   int   strideAccess;           // increment formula for the index variable (when numRepeats>1)
   int   globalBuffPadding;      // padding for the global array (unused)
   bool  useLocalMem;            // read/write data from local buffer?
   int   localBuffSize;          // size of local buffer
   int   localBuffPadding;       // padding for local buffer (to avoid costly modulo inside the kernel loop)
   int   numRepeats;             // number of loop iterations (>= 1)
   int   numUnrolls;             // number of loop unrolls (>= 1)
   int   minGroupSize;           // minimum local work group size
   int   maxGroupSize;           // maximum local work group size (0 to use the card's maximum)
   unsigned int opFlags;         // indicates if kernel applies to READ and/or WRITE operations
} bTestsGPU[] = {
  {"GlobalMemoryCoalesced", "__global", "__global", "s", "gid", false, -1, 0, false, 0, 0, 1024, 16, 32, 0, OP_MEM_READ|OP_MEM_WRITE},
  {"GlobalMemoryUnit", "__global", "__global", "s", "gid*1024", false, 1, 0, false, 0, 0, 512, 16, 32, 0, OP_MEM_READ|OP_MEM_WRITE},
  {"ConstantMemoryCoalesced", "__global const", "__global", "s", "gid", false, -1, 0, false, 0, 0, 1024, 16, 32, 0, OP_MEM_READ},
  {"LocalMemory", "__global const", "__global", "s", "tid", false, 1, 0, true, 2048, 0, 3000, 16, 32, 0, OP_MEM_READ|OP_MEM_WRITE},
  {0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0}
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
//    Jeremy Meredith, Tue Nov 23 11:18:06 EST 2010
//    Removing normalized option for now as it may have been accessing memory
//    in an unintended pattern and leading to unexpected performance results.
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
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
//   kName: kernel name
//   opt: operation type (READ, WRITE)
//   etype: data type (int, float, etc)
//
// Returns:  nothing
//
// Programmer: Gabriel Marin
// Creation: July 08, 2009
//
// Modifications:
//   Gabriel Marin, 06/09/2010: Use more device specific information when
//   generating the OpenCL kernels. Change memory access patterns to
//   avoid cache reuse.
//
//   Gabriel Marin, 01/14/2011: Increase slightly the number of avaialble
//   warps per SM, increase utilization slightly for local memory benchmarks
//   by making sure that the local buffer size is at most half the size
//   of the local memory. Change local memory access pattern to minimize
//   bank conflicts.
//
// ****************************************************************************
void
generateKernel (ostringstream &oss,
                struct _benchmark_type &test,
                string &kName,
                int opt,
                int etype,
                size_t localMem,
                size_t numThreads,
                size_t memSizePerThread)
{
    int u;
    int elemSize;
    long localMemSize = 0;
    if (etype==INT_TYPE)
       elemSize = sizeof(int);
    else
       elemSize = sizeof(float);
    if (test.useLocalMem){
       if (localMem > 0)
       {
          long maxLocal = localMem/elemSize;
          //// find the largest power of 2, strictly less than localMemSize
          //localMemSize = lround(exp2(floor(log2(maxLocal-1))));

          // find the largest power of 2, less or equal than half of localMemSize
          localMemSize = lround(exp2(floor(log2(maxLocal>>1))));
       }
       else
          localMemSize = test.localBuffSize;
    }
    oss << std::string("__kernel void ") << kName << "(";
    if (opt == OP_MEM_READ)
        oss << test.sourceType << " "
            << etypeNames[etype] << " *data, ";
    oss << test.destType << " " << etypeNames[etype]
        << " *output";
    oss << std::string(", int size)\n") <<
        "{\n" <<
        "    int gid = get_global_id(0), num_thr = get_global_size(0), grpid=get_group_id(0), j = 0;\n";
    if (test.numRepeats > 1)
        oss << "    " << etypeNames[etype] << " sum = 0;\n";
    if (test.useLocalMem)
    {
        oss << "    int tid=get_local_id(0), localSize=get_local_size(0), litems="
            << localMemSize << "/localSize, goffset=localSize*grpid+tid*litems;\n";
    }
    oss << "    int " << test.indexVar << " = ";
    if (test.scaleInitFormula)
       oss << "(" << test.indexVarInit << ")*" << memSizePerThread << ";\n";
    else
       oss << test.indexVarInit << ";\n";
    if (test.useLocalMem)
        oss << std::string("    __local ") << etypeNames[etype]
            << " lbuf[" << localMemSize << "];\n";

    if (test.useLocalMem && opt==OP_MEM_READ)
        oss << "    for ( ; j<litems && j<(size-goffset) ; ++j)\n"
            << "       lbuf[tid*litems+j] = data[goffset+j];\n"
            << "    for (int i=0 ; j<litems ; ++j,++i)\n"
            << "       lbuf[tid*litems+j] = data[i];\n"
            << "    barrier(CLK_LOCAL_MEM_FENCE);\n";

    int usedStride = 0;
    if (test.strideAccess < 0)
       usedStride = numThreads;
    else
       usedStride = test.strideAccess;
    if (opt == OP_MEM_READ)
    {
        if (test.numRepeats > 1)
        {
            oss << "    for (j=0 ; j<"
                << test.numRepeats << " ; ++j) {\n";
        }
        for (u=0 ; u<test.numUnrolls ; ++u)
        {
            if (test.useLocalMem)
                oss << "       " << etypeNames[etype] << " a"
                    << u << " = lbuf[("
                    << test.indexVar << "+"
                    << u*usedStride
                    << ")&(" << (localMemSize-1) << ")];\n";
            else
                oss << "       " << etypeNames[etype] << " a"
                    << u << " = data[("
                    << test.indexVar << "+"
                    << u*usedStride
                    << ")&(size-1)];\n";
        }
        oss << "       sum += a0";
        for (u=1 ; u<test.numUnrolls ; ++u)
            oss << "+a" << u;
        oss << ";\n";
        if (test.numRepeats > 1)
        {
            oss << "       " << test.indexVar << " = ("
                << test.indexVar << "+"
                << test.numUnrolls * usedStride
                << ")&(";
            if (test.useLocalMem)
               oss << (localMemSize-1);
            else
               oss << "size-1";
            oss << ");\n"
                << "    }\n"
                << "    output[gid] = sum;\n";
        }
    }
    else
    {   // it is a MEM_WRITE kernel
        if (test.numRepeats > 1)
        {
            oss << "    for (j=0 ; j<" << test.numRepeats
                << " ; ++j) {\n";
        }
        for (u=0 ; u<test.numUnrolls ; ++u)
        {
            if (test.useLocalMem)
                oss << "       lbuf[("
                    << test.indexVar << "+"
                    << u*usedStride
                    << ")&(" << (localMemSize-1) << ")] = gid;\n";
            else
                oss << "       output[("
                    << test.indexVar << "+"
                    << u*usedStride
                    << ")&(size-1)] = gid;\n";
        }
        if (test.numRepeats > 1)
        {
            oss << "       " << test.indexVar << " = ("
                << test.indexVar << "+"
                << test.numUnrolls * usedStride
                << ")&(";
            if (test.useLocalMem)
               oss << (localMemSize-1);
            else
               oss << "size-1";
            oss << ");\n"
                << "    }\n";
        }
    }
    if (test.useLocalMem && opt==OP_MEM_WRITE)
    {
        oss << "    barrier(CLK_LOCAL_MEM_FENCE);\n"
            << "    for (j=0 ; j<litems ; ++j)\n"
            << "       output[gid] = lbuf[tid];\n";
    }
    oss << "}\n";
}

inline int min(int x, int y) {
    if (x < y)
        return x;
    else
        return y;
}


// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes a series of memory benchmarks for OpenCL devices.
//   OpenCL kernels are auto-generated based on the values in the
//   _benchmark_type structures.
//   The benchmark tests read and write bandwidths for global, constant,
//   and local OpenCL device memory. Memory benchmarks are
//   executed for different work group sizes.
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
// Creation: July 08, 2009
//
// Modifications:
//   Gabriel Marin, 06/09/2010: Use more device specific information when
//   generating the OpenCL kernels. Change memory access patterns to
//   avoid cache reuse.
//
//   Gabriel Marin, 01/14/2011: Increase slightly the number of avaialble
//   warps per SM, increase utilization slightly for local memory benchmarks
//   by making sure that the local buffer size is at most half the size
//   of the local memory. Change local memory access pattern to minimize
//   bank conflicts.
//
// ****************************************************************************
void RunBenchmark(cl_device_id dev,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    const bool waitForEvents = true;
    bool verbose = op.getOptionBool("verbose");
    bool quiet = op.getOptionBool("quiet");
    int npasses = op.getOptionInt("passes");

    // 1k through 8M bytes
    int minGroupSize = 32;
    size_t maxGroupSize = 0;

    int numSMs = getMaxComputeUnits(dev);
    cl_device_type type;
    clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    if (type == CL_DEVICE_TYPE_CPU)
    {
       minGroupSize = 256;
    }

    int memSize = 64*1024*1024;  // 64MB buffer
    const long availMem = findAvailBytes(dev);
    while (memSize*2 > availMem)
       memSize >>= 1;   // keep it a power of 2

    const int numWordsInt = memSize / sizeof(int);
    const int numWordsFloat = memSize / sizeof(float);

    size_t numWarps = numSMs * 16;  // use 8 times as many warps as the number of compute units
    size_t globalWorkSize = numWarps * 32;  // each warp has 32 threads
    int elemSize;

    // initialize host memory
    int *hostMemInt = new int[numWordsInt];
    float *hostMemFloat = new float[numWordsFloat];
    int *outMemInt = new int[numWordsInt];
    float *outMemFloat = new float[numWordsFloat];
    srand48(8650341L);
    for (int i=0 ; i<numWordsInt ; ++i)
    {
        hostMemInt[i] = (int)floor(drand48()*numWordsInt);
    }
    for (int i=0 ; i<numWordsFloat ; ++i)
    {
        hostMemFloat[i] = (float)(drand48()*numWordsFloat);
    }

    // Allocate some device memory
    int err;
    char sizeStr[128];
    bool addTypeSuffix = false;
    cl_ulong localMemSize = getLocalMemSize(dev);

    struct _benchmark_type *bTests = bTestsGPU;

    int opt, etype;
    int totalRuns = 0;

    // compute how many runs we have to make
    for (opt=OP_MEM_READ ; opt<NUM_MEMORY_OPERATIONS ; ++opt)
    {
        // for each data type; currently only floating point type is used because
        // there were no major differences in the performance results with
        // integer and floating point data types.
        for (etype=FLOAT_TYPE ; etype<NUM_ELEM_TYPES ; ++etype)
        {
            int bIdx = 0;
            while ((bTests!=0) && (bTests[bIdx].name!=0))
            {
                if (! (bTests[bIdx].opFlags & opt))
                {
                    ++ bIdx;
                    continue;
                }
                std::string kName = std::string(opNames[opt]) +
                          bTests[bIdx].name;
                std::ostringstream oss;
                generateKernel (oss, bTests[bIdx], kName, opt, etype, localMemSize, 32, 1024);
                std::string kernelCode(oss.str());

                const char* progSource[] = {kernelCode.c_str()};
                cl_program prog = clCreateProgramWithSource(ctx, 1,
                                  progSource, NULL, &err);
                CL_CHECK_ERROR(err);

                // Compile the program
                err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
                CL_CHECK_ERROR(err);

                // Extract out memory kernels
                cl_kernel kernel_mem = clCreateKernel(prog,
                              kName.c_str(), &err);
                CL_CHECK_ERROR(err);

                if (!maxGroupSize)
                {
                    maxGroupSize = min(getMaxWorkGroupSize(dev), 512);
                }
                if (bTests[bIdx].minGroupSize < minGroupSize)
                    bTests[bIdx].minGroupSize = minGroupSize;
                if (bTests[bIdx].minGroupSize > maxGroupSize)
                    bTests[bIdx].minGroupSize = maxGroupSize;
                if (!bTests[bIdx].maxGroupSize || bTests[bIdx].maxGroupSize > maxGroupSize)
                    bTests[bIdx].maxGroupSize = maxGroupSize;

                // Run the kernel for each group size
                for (int wsize=bTests[bIdx].minGroupSize ; wsize<=bTests[bIdx].maxGroupSize ; wsize*=2)
                {
                    totalRuns += npasses;
                }
                err = clReleaseKernel (kernel_mem);
                CL_CHECK_ERROR(err);
                err = clReleaseProgram (prog);
                CL_CHECK_ERROR(err);

                bIdx += 1;
            }
        }
    }
    ProgressBar pb(totalRuns);
    if (!verbose && !quiet)
        pb.Show(stdout);

    globalWorkSize = numSMs * maxGroupSize * 5;

    // for each type of memory operation
    for (opt=OP_MEM_READ ; opt<NUM_MEMORY_OPERATIONS ; ++opt)
    {
        // for each data type; currently only floating point type is used because
        // there were no major differences in the performance results with
        // integer and floating point data types.
        for (etype=FLOAT_TYPE ; etype<NUM_ELEM_TYPES ; ++etype)
        {
            cl_mem mem1, mem2;
            ElemType et = (ElemType)etype;
            switch (et)
            {
                case INT_TYPE:
                {
                    elemSize = sizeof(int);
                    mem1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 sizeof(cl_int)*(numWordsInt),
                                 NULL, &err);
                    CL_CHECK_ERROR(err);
                    Event evDownloadPrime("DownloadPrime");
                    err = clEnqueueWriteBuffer(queue, mem1, false, 0,
                                 (numWordsInt)*sizeof(int),
                                 hostMemInt,
                                 0, NULL,
                                 &evDownloadPrime.CLEvent());
                    CL_CHECK_ERROR(err);
                    err = clWaitForEvents(1, &evDownloadPrime.CLEvent());
                    CL_CHECK_ERROR(err);

                    mem2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 sizeof(cl_int)*(numWordsInt),
                                 NULL, &err);
                    CL_CHECK_ERROR(err);
                    Event evDownloadPrime2("DownloadPrime");
                    err = clEnqueueWriteBuffer(queue, mem2, false, 0,
                                 (numWordsInt)*sizeof(int),
                                 hostMemInt,
                                 0, NULL,
                                 &evDownloadPrime2.CLEvent());
                    CL_CHECK_ERROR(err);
                    err = clWaitForEvents(1, &evDownloadPrime2.CLEvent());
                    CL_CHECK_ERROR(err);
                }
                break;

                case FLOAT_TYPE:
                {
                    elemSize = sizeof(float);
                    mem1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 sizeof(cl_float)*(numWordsFloat),
                                 NULL, &err);
                    CL_CHECK_ERROR(err);
                    Event evDownloadPrime("DownloadPrime");
                    err = clEnqueueWriteBuffer(queue, mem1, false, 0,
                                 (numWordsFloat)*sizeof(float),
                                 hostMemFloat,
                                 0, NULL,
                                 &evDownloadPrime.CLEvent());
                    CL_CHECK_ERROR(err);
                    err = clWaitForEvents(1, &evDownloadPrime.CLEvent());
                    CL_CHECK_ERROR(err);

                    mem2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 sizeof(cl_float)*(numWordsFloat),
                                 NULL, &err);
                    CL_CHECK_ERROR(err);
                    Event evDownloadPrime2("DownloadPrime");
                    err = clEnqueueWriteBuffer(queue, mem2, false, 0,
                                 (numWordsFloat)*sizeof(float),
                                 hostMemFloat,
                                 0, NULL,
                                 &evDownloadPrime2.CLEvent());
                    CL_CHECK_ERROR(err);
                    err = clWaitForEvents(1, &evDownloadPrime2.CLEvent());
                    CL_CHECK_ERROR(err);
                }
                break;

                default:
                break;
            }

            int bIdx = 0;
            while ((bTests!=0) && (bTests[bIdx].name!=0))
            {
                if (! (bTests[bIdx].opFlags & opt))
                {
                    ++ bIdx;
                    continue;
                }
                std::string kName = std::string(opNames[opt]) +
                          bTests[bIdx].name;
                if (addTypeSuffix)
                    kName = kName + "_" + etypeNames[etype];
                std::ostringstream oss;
                generateKernel (oss, bTests[bIdx], kName, opt, etype, localMemSize, globalWorkSize, memSize/elemSize/globalWorkSize);
                std::string kernelCode(oss.str());

                if (verbose)
                    cout << "Code for kernel " << kName
                         << ":\n" + kernelCode << endl;

                const char* progSource[] = {kernelCode.c_str()};
                cl_program prog = clCreateProgramWithSource(ctx, 1,
                                  progSource, NULL, &err);
                CL_CHECK_ERROR(err);

                // Compile the program
                err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
                CL_CHECK_ERROR(err);

                // check if we have to dump the PTX
                char* dumpPtx = getenv("DUMP_PTX");
                if (dumpPtx && !strcmp(dumpPtx, "1"))
                    // must dump the PTX
                    dumpPTXCode(ctx, prog, kName.c_str());

                // Extract out memory kernels
                cl_kernel kernel_mem = clCreateKernel(prog,
                              kName.c_str(), &err);
                CL_CHECK_ERROR(err);

                int argIdx = 0;
                if (opt == OP_MEM_READ)
                {
                    err = clSetKernelArg(kernel_mem, argIdx,
                                sizeof(cl_mem),
                                (void*)&mem1);
                    CL_CHECK_ERROR(err);
                    ++ argIdx;
                }
                err = clSetKernelArg(kernel_mem, argIdx,
                                sizeof(cl_mem),
                                (void*)&mem2);
                CL_CHECK_ERROR(err);
                ++ argIdx;
                if (et == INT_TYPE)
                   err = clSetKernelArg(kernel_mem, argIdx,
                                sizeof(cl_int),
                                (void*)&numWordsInt);
                else
                   err = clSetKernelArg(kernel_mem, argIdx,
                                sizeof(cl_int),
                                (void*)&numWordsFloat);
                CL_CHECK_ERROR(err);
                ++ argIdx;

                // find the maximum allowed group size
                if (!maxGroupSize)
                {
                    maxGroupSize = min(getMaxWorkGroupSize(dev), 512);
                }
                if (bTests[bIdx].minGroupSize < minGroupSize)
                    bTests[bIdx].minGroupSize = minGroupSize;
                if (bTests[bIdx].minGroupSize > maxGroupSize)
                    bTests[bIdx].minGroupSize = maxGroupSize;
                if (!bTests[bIdx].maxGroupSize || bTests[bIdx].maxGroupSize > maxGroupSize)
                    bTests[bIdx].maxGroupSize = maxGroupSize;

                // Run the kernel for each group size
                for (int wsize=bTests[bIdx].minGroupSize ; wsize<=bTests[bIdx].maxGroupSize ; wsize*=2)
                {
                    size_t localWorkSize = wsize;
                    numWarps = globalWorkSize / localWorkSize;
//                    globalWorkSize = numWarps*localWorkSize;

                    if (verbose)
                        cout << ">> running the " << kName
                             << " kernel, globalWorkSize=" << globalWorkSize
                             << ", localWorkSize=" << localWorkSize
                             << " and number of groups=" << numWarps << "\n";

                    for (int pas=0 ; pas<npasses ; ++pas)
                    {
                        Event evKernel(kName.c_str());
                        err = clEnqueueNDRangeKernel(queue,
                                    kernel_mem, 1, NULL,
                                    &globalWorkSize, &localWorkSize,
                                    0, NULL, &evKernel.CLEvent());
                        CL_CHECK_ERROR(err);

                        // Wait for the kernel to finish
                        err = clWaitForEvents(1, &evKernel.CLEvent());
                        CL_CHECK_ERROR(err);

                        // Read the result device memory back to the host
                        Event evReadback("Readback");
                        if (et == INT_TYPE)
                        {
                            err = clEnqueueReadBuffer(queue, mem2, false, 0,
                                     numWordsInt*sizeof(int), outMemInt,
                                     0, NULL, &evReadback.CLEvent());
                        } else
                        {
                            err = clEnqueueReadBuffer(queue, mem2, false, 0,
                                     numWordsFloat*sizeof(float), outMemFloat,
                                     0, NULL, &evReadback.CLEvent());
                        }
                        CL_CHECK_ERROR(err);

                        err = clWaitForEvents(1, &evReadback.CLEvent());
                        CL_CHECK_ERROR(err);

                        evKernel.FillTimingInfo();

                        double bdwth = ((double)globalWorkSize*bTests[bIdx].numRepeats*bTests[bIdx].numUnrolls*elemSize)
                                 / double(evKernel.SubmitEndRuntime());
                        sprintf (sizeStr, "GrpSize:%03d", wsize);
                        resultDB.AddResult(kName.c_str(),
                                 sizeStr, "GB/s", bdwth);

                        // update progress bar
                        pb.addItersDone();
                        if (!verbose && !quiet)
                            pb.Show(stdout);
                    }
                }

                err = clReleaseKernel (kernel_mem);
                CL_CHECK_ERROR(err);
                err = clReleaseProgram (prog);
                CL_CHECK_ERROR(err);

                bIdx += 1;
            }  // for each kernel configuration
            err = clReleaseMemObject(mem1);
            CL_CHECK_ERROR(err);
            err = clReleaseMemObject(mem2);
            CL_CHECK_ERROR(err);
        }  // for each data type
    }  // for each memory operation
    fprintf (stdout, "\n\n");

    delete[] hostMemInt;
    delete[] hostMemFloat;
    delete[] outMemInt;
    delete[] outMemFloat;

    TestImageMemory(ctx, queue, resultDB, op);
}


void
printImageNoResults(unsigned int passes, unsigned int nsizes, unsigned int sIdx,
           const unsigned int sizes[], ResultDatabase &resultDB)
{
   int i, j;
   char sizeStr[256];

   for (j=sIdx ; j<nsizes ; ++j)
   {
       sprintf(sizeStr, "% 6dkB", sizes[j]);
       for (i=0 ; i<passes ; ++i)
       {
            // use the same names as the cuda versions
            resultDB.AddResult("TextureRepeatedLinearAccess", sizeStr, "GB/sec", FLT_MAX);
            resultDB.AddResult("TextureRepeatedRandomAccess", sizeStr, "GB/sec", FLT_MAX);
            resultDB.AddResult("TextureRepeatedCacheHit",     sizeStr, "GB/sec", FLT_MAX);
       }
   }
}

// ****************************************************************************
// Function: TestImageMemory
//
// Purpose:
//   Executes a series of image memory benchmarks for OpenCL devices.  Three
//   access patterns are used: a simple linear read, repeated reads to a small
//   region of data designed to fit into the texture cache (assuming cache is at
//   least 4kb in size), and a "random" access pattern.  Results are reported to
//   the result DB in terms of bandwidth.
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
// Creation: December 14, 2009
//
// Modifications:
//    Gabriel Marin, November 18, 2010
//        - change image from 1D to 2D
//        - query device properties to understand maximum image size and maximum thread group size
//        - change image type from 1 channel to 4 channels
//        - add option for normalized or unnormalized coordinates
//        - report special value if image type is not supported.
//
//    Jeremy Meredith, Tue Nov 23 11:18:06 EST 2010
//    Removing normalized option for now as it may have been accessing memory
//    in an unintended pattern and leading to unexpected performance results.
//
// ****************************************************************************
void TestImageMemory(cl_context ctx,
                     cl_command_queue queue,
                     ResultDatabase &resultDB,
                     OptionParser &op) {

    int err = 0;

    // verify that the device actually supports image memory
    cl_device_id device_id;
    cl_bool deviceSupportsImages;
    size_t maxWidth = 0, maxHeight = 0;
    size_t globalWorkSize[2], localWorkSize[2];
    // Number of times to repeat each test
    const unsigned int passes = op.getOptionInt("passes");

    // Sizes of textures tested (in kb)
    const unsigned int nsizes = 5;
    const unsigned int sizes[] = { 16, 64, 256, 1024, 4096 };


    err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device_id),
                                &device_id, NULL);
    CL_CHECK_ERROR(err);

    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT,
            sizeof(deviceSupportsImages), &deviceSupportsImages, NULL);
    CL_CHECK_ERROR(err);

    cl_uint numDimensions = 0;
    clGetDeviceInfo (device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                             sizeof(cl_uint), &numDimensions, NULL);

    if (!deviceSupportsImages || numDimensions<2)
    {
        std::cout << "Images are not supported for device 0x"
            << std::hex << device_id
            << std::endl;
        printImageNoResults(passes, nsizes, 0, sizes, resultDB);
        return;
    }

    // device supports image memory, so proceed with the test
    cout << "Now testing image memory.\n";
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH,
            sizeof(size_t), &maxWidth, NULL);
    CL_CHECK_ERROR(err);
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
            sizeof(size_t), &maxHeight, NULL);
    CL_CHECK_ERROR(err);
    size_t maxMemSize = maxWidth*maxHeight*sizeof(float)*4;
    if (sizes[nsizes-1]*1024 > maxMemSize)  // device does not support all memory sizes
    {
        std::cout << "Not all image sizes are supported by device 0x"
            << std::hex << device_id
            << std::endl;
        printImageNoResults(passes, nsizes, 0, sizes, resultDB);
        return;
    }

    // Number of loop repetitions in each kernel
    int numSMs = getMaxComputeUnits(device_id);
    const unsigned int kernelRepFactor = 1024*numSMs/16;

    size_t *maxWorkSizes = new size_t[numDimensions];
    clGetDeviceInfo (device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                       sizeof(size_t)*numDimensions, maxWorkSizes, NULL);

    cl_sampler sampler;
    cl_device_type type;
    clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    sampler = clCreateSampler (ctx, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &err);
    if (err != CL_SUCCESS)
    {
        cout << "Device does not support required sampler type, skipping test\n";
        printImageNoResults(passes, nsizes, 0, sizes, resultDB);
        return;
    }

    // Create Kernels
    const char* kernelSource =
            "__kernel void readImg(int n, __global float *d_out,          "
            "    __read_only image2d_t img, sampler_t samp, int w, int h) "
            "{                                                            "
            "    int2 ridx = (int2)(get_global_id(0),get_global_id(1));   "
            "    int idx = ridx.x*get_global_size(1) + ridx.y;"
            "    float sum = 0.0f;                                        "
            "    w = w-1; "
            "    for (int i = 0; i < n; i++)                              "
            "    {                                                        "
            "        float4 x = read_imagef(img, samp, ridx); "
            "        ridx.x = (ridx.x+1)&(w); "
            "        sum += x.x;                                          "
            "    }                                                        "
            "    d_out[idx] = sum;                                        "
            "} "
            "__kernel void readInCache(int n, __global float *d_out,      "
            "    __read_only image2d_t img, sampler_t samp) "
            "{"
            "    int2 ridx = (int2)(get_global_id(0),get_global_id(1));   "
            "    int idx = ridx.x*get_global_size(1) + ridx.y;"
            "    float sum = 0.0f;                                        "
            "    for (int i = 0; i < n; i++)                              "
            "    {                                                        "
            "        float4 x = read_imagef(img, samp, ridx); "
            "        sum += x.x;                                          "
            "    }                                                        "
            "    d_out[idx] = sum;                                        "
            "}                                                            "
            "__kernel void readRand(int n, __global float *d_out,         "
            "    __read_only image2d_t img, sampler_t samp, int w, int h) "
            "{ "
            "    int2 ridx = (int2)(get_global_id(0),get_global_id(1));   "
            "    int idx = ridx.x*get_global_size(1) + ridx.y;"
            "    float sum = 0.0f; "
            "    w = w-1; h = h-1; "
            "    for (int i = 0; i < n; i++) "
            "    { "
            "        float4 x = read_imagef(img, samp, ridx); "
            "        ridx.x = (ridx.x*3+29)&(w); "
            "        ridx.y = (ridx.y*5+11)&(h); "
            "        sum += x.x; "
            "    } "
            "    d_out[idx] = sum; "
            "} ";
    // Create program option from the source string
    cl_program prog = clCreateProgramWithSource(ctx, 1,
            &kernelSource, NULL, &err);
    CL_CHECK_ERROR(err);

    // Compile the program
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    CL_CHECK_ERROR(err);

    // Extract out kernels
    cl_kernel linear = clCreateKernel(prog, "readImg", &err);
    CL_CHECK_ERROR(err);

    cl_kernel cache = clCreateKernel(prog, "readInCache", &err);
    CL_CHECK_ERROR(err);

    cl_kernel rand = clCreateKernel(prog, "readRand", &err);
    CL_CHECK_ERROR(err);

    // Current NVIDIA OCL Implementation seems to have a problem with
    // local sizes above 64, so make that an upper bound for now (12/14)
    localWorkSize[0] = min(16, maxWorkSizes[0]);
    localWorkSize[1] = min(8, maxWorkSizes[1]);

    for (int j = 0; j < nsizes; j++)
    {
        cout << "Benchmarking Image Memory, Test: " << j+1 << " / 5\n";
        const unsigned int size      = 1024 * sizes[j];
        const unsigned int numFloat  = size / sizeof(float);
        size_t numFloat4 = size / (sizeof(float)*4);
        size_t width, height;

        // Image memory sizes should be power of 2.
        size_t sizeLog = lround(log2(numFloat4));
        height = 1 << (sizeLog >> 1);  // height is the smaller size
        if (height>maxHeight) height = maxHeight;
        width = numFloat4 / height;
        if (width>maxWidth) {
            cout << "Image size is not supported, though initial test passed." << endl;
            err = clReleaseKernel(linear);
            CL_CHECK_ERROR(err);
            err = clReleaseKernel(cache);
            CL_CHECK_ERROR(err);
            err = clReleaseKernel(rand);
            CL_CHECK_ERROR(err);
            err = clReleaseProgram(prog);
            CL_CHECK_ERROR(err);
            err = clReleaseSampler(sampler);
            CL_CHECK_ERROR(err);
            printImageNoResults(passes, nsizes, j, sizes, resultDB);
            return;
        }
        //cout << "Buf size=" << size << ", imgWidth=" << width << ", imgHeight=" << height << endl;

        float *h_in = new float[numFloat];
        float *h_out = new float[numFloat4];

        // Fill input data with some pattern
        for (unsigned int i = 0; i < numFloat; i++)
        {
            h_in[i] = (float)i;
            if (i < numFloat4)
            {
                h_out[i] = 0.0f;
            }
        }

        // Create dev memory for output
        cl_mem d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                sizeof(float)*numFloat4, NULL, &err);
        CL_CHECK_ERROR(err);

        // Set up opencl image format
        cl_image_format img_format;
        img_format.image_channel_order = CL_RGBA;
        img_format.image_channel_data_type = CL_FLOAT;

        // Create opencl image
        cl_mem d_img = clCreateImage2D(ctx,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &img_format,
                width, height, 0, h_in, &err);

        // Stop the test if the device supports images, but not the
        // specified format
        if (err == CL_IMAGE_FORMAT_NOT_SUPPORTED ||
            err == CL_INVALID_IMAGE_SIZE)
        {
           cout << "Image format or size not supported, skipping test\n";
           delete[] h_in;
           delete[] h_out;
           err = clReleaseMemObject(d_out);
           CL_CHECK_ERROR(err);
           err = clReleaseKernel(linear);
           CL_CHECK_ERROR(err);
           err = clReleaseKernel(cache);
           CL_CHECK_ERROR(err);
           err = clReleaseKernel(rand);
           CL_CHECK_ERROR(err);
           err = clReleaseProgram(prog);
           CL_CHECK_ERROR(err);
           err = clReleaseSampler(sampler);
           CL_CHECK_ERROR(err);
           printImageNoResults(passes, nsizes, j, sizes, resultDB);
           return;
        }

        Event evKernel("texture kernel");

        globalWorkSize[0] = width;
        globalWorkSize[1] = height;

        // Set Kernel Arguments
        err = clSetKernelArg(linear, 0, sizeof(cl_int),
                (void*)&kernelRepFactor);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(linear, 1, sizeof(cl_mem), (void*)&d_out);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(linear, 2, sizeof(cl_mem), (void*)&d_img);
        CL_CHECK_ERROR(err);
           err = clSetKernelArg(linear, 3, sizeof(cl_sampler), (void*)&sampler);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(linear, 4, sizeof(cl_int), (void*)&width);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(linear, 5, sizeof(cl_int), (void*)&height);
        CL_CHECK_ERROR(err);

        err = clSetKernelArg(cache, 0, sizeof(cl_int),
                (void*)&kernelRepFactor);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(cache, 1, sizeof(cl_mem), (void*)&d_out);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(cache, 2, sizeof(cl_mem), (void*)&d_img);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(cache, 3, sizeof(cl_sampler), (void*)&sampler);
        CL_CHECK_ERROR(err);

        err = clSetKernelArg(rand, 0, sizeof(cl_int),
                (void*)&kernelRepFactor);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(rand, 1, sizeof(cl_mem), (void*)&d_out);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(rand, 2, sizeof(cl_mem), (void*)&d_img);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(rand, 3, sizeof(cl_sampler), (void*)&sampler);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(rand, 4, sizeof(cl_int), (void*)&width);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(rand, 5, sizeof(cl_int), (void*)&height);
        CL_CHECK_ERROR(err);

        for (int p = 0; p < passes; p++)
        {
            // Test 1: Repeated Linear Access
            double t = 0;

            // read texels from texture
            err = clEnqueueNDRangeKernel(queue, linear, 2, NULL, globalWorkSize,
                    localWorkSize, 0, NULL, &evKernel.CLEvent());
            clFinish(queue);
            CL_CHECK_ERROR(err);

            evKernel.FillTimingInfo();
            t = evKernel.SubmitEndRuntime();

            // Calculate speed in GB/s
            double speed = (double)kernelRepFactor * (double)(size) / t;

            char sizeStr[256];
            sprintf(sizeStr, "% 6dkB", size / 1024);
            resultDB.AddResult("TextureRepeatedLinearAccess", sizeStr, "GB/sec",
                    speed);

            // Verify results
            err = clEnqueueReadBuffer(queue, d_out, true, 0,
                    numFloat4 * sizeof(float), h_out, 0, NULL, NULL);
            CL_CHECK_ERROR(err);

            // Test 2 Repeated Cache Access
            err = clEnqueueNDRangeKernel(queue,
                    cache, 2, NULL, globalWorkSize,
                    localWorkSize, 0, NULL, &evKernel.CLEvent());
            CL_CHECK_ERROR(err);

            // Wait for the kernel to finish
            err = clWaitForEvents(1, &evKernel.CLEvent());
            CL_CHECK_ERROR(err);

            evKernel.FillTimingInfo();
            t = evKernel.SubmitEndRuntime();

            // Calculate speed in GB/s
            speed = (double)kernelRepFactor * (double)size / t;
            sprintf(sizeStr, "% 6dkB", size / 1024);
            resultDB.AddResult("TextureRepeatedCacheHit", sizeStr, "GB/sec",
                    speed);

            // Verify results
            err = clEnqueueReadBuffer(queue, d_out, true, 0,
                    numFloat4 * sizeof(float), h_out, 0, NULL, NULL);
            CL_CHECK_ERROR(err);

            // Test 3 Repeated "Random" Access
            err = clEnqueueNDRangeKernel(queue,
                    rand, 2, NULL, globalWorkSize,
                    localWorkSize, 0, NULL, &evKernel.CLEvent());
            clFinish(queue);
            CL_CHECK_ERROR(err);

            evKernel.FillTimingInfo();
            t = evKernel.SubmitEndRuntime();

            // Calculate speed in GB/s
            speed = (double)kernelRepFactor * (double)size / t;
            sprintf(sizeStr, "% 6dkB", size / 1024);
            resultDB.AddResult("TextureRepeatedRandomAccess", sizeStr, "GB/sec",
                    speed);
        }
        delete[] h_in;
        delete[] h_out;
        err = clReleaseMemObject(d_out);
        CL_CHECK_ERROR(err);
        err = clReleaseMemObject(d_img);
        CL_CHECK_ERROR(err);
    }
    err = clReleaseSampler(sampler);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(linear);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(cache);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(rand);
    CL_CHECK_ERROR(err);
    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
}
