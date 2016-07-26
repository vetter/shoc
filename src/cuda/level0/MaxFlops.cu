#include "cudacommon.h"
#include <stdio.h>
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "ProgressBar.h"
#include "Utility.h"

// Forward Declarations for benchmark kernels
__global__ void    MAddU(float *target, float val1, float val2);
__global__ void MulMAddU(float *target, float val1, float val2);
__global__ void MAddU_DP(double *target, double val1, double val2);
__global__ void MulMAddU_DP(double *target, double val1, double val2);

// Add kernels
template <class T> __global__ void Add1(T *data, int nIters, T v);
template <class T> __global__ void Add2(T *data, int nIters, T v);
template <class T> __global__ void Add4(T *data, int nIters, T v);
template <class T> __global__ void Add8(T *data, int nIters, T v);

// Mul kernels
template <class T> __global__ void Mul1(T *data, int nIters, T v);
template <class T> __global__ void Mul2(T *data, int nIters, T v);
template <class T> __global__ void Mul4(T *data, int nIters, T v);
template <class T> __global__ void Mul8(T *data, int nIters, T v);

// MAdd kernels
template <class T> __global__ void MAdd1(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MAdd2(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MAdd4(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MAdd8(T *data, int nIters, T v1, T v2);

// MulMAdd kernels
template <class T> __global__ void MulMAdd1(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MulMAdd2(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MulMAdd4(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MulMAdd8(T *data, int nIters, T v1, T v2);


// Forward Declarations
// execute simple precision and double precision versions of the benchmarks
template <class T> void
RunTest(ResultDatabase &resultDB, int npasses, int verbose, int quiet,
        float repeatF, ProgressBar &pb, const char* precision);

// Block size to use in measurements
#define BLOCK_SIZE_SP 256
#define BLOCK_SIZE_DP 128

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
// Creation: December 11, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   This benchmark measures the max floating point capability of a gpu using
//   a highly unrolled kernel with a large number of floating point operations.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: September 08, 2009
//
// Modifications:
//    Jeremy Meredith, Fri May 14 11:23:10 EDT 2010
//    Made double precision a copy of SP, with a few tweaks.
//    Allow any capability at least 1.3 or 2.0 to use double.
//
//    Gabriel Marin, Thu Jan 13, 2010
//    Add the auto-generated kernels from the OpenCL implementation.
//    DP / SP implemented as templates for the new kernels.
//    Add text progress bar.
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{
    bool verbose = op.getOptionBool("verbose");
    bool quiet = op.getOptionBool("quiet");
    const unsigned int passes = op.getOptionInt("passes");

    // Test to see if this device supports double precision
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    bool doDouble = false;
    if ((deviceProp.major == 1 && deviceProp.minor >= 3) ||
            (deviceProp.major >= 2))
    {
        doDouble = true;
    }

    // determine the speed of the device first. This determines the number of
    // iterations for all kernels.
    const unsigned int halfBufSize = 1024*1024;
    unsigned int halfNumFloats = halfBufSize / sizeof(float), numFloats = 2*halfNumFloats;
    float *gpu_mem, *hostMem;
    hostMem = new float[numFloats];
    cudaMalloc((void**)&gpu_mem, halfBufSize*2);
    CHECK_CUDA_ERROR();

    // Initialize host data, with the first half the same as the second
    for (int j=0; j<halfNumFloats; ++j)
    {
        hostMem[j] = hostMem[numFloats-j-1] = (float)(drand48()*10.0);
    }

    // Variables used for timing
    float t = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CHECK_CUDA_ERROR();

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, halfBufSize*2, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Thread block configuration
    dim3 threads(BLOCK_SIZE_SP,1,1);
    dim3 blocks((numFloats)/BLOCK_SIZE_SP,1,1);

    // Decrease block size for devices with lower compute
    // capability.  Avoids an out of resources error
    if ((deviceProp.major == 1 && deviceProp.minor <= 2))
    {
        threads.x = 128;
        blocks.x  = (numFloats)/128;
    }

    // Benchmark the MulMAdd2 kernel to compute a scaling factor.
    t = 0.0f;
    cudaEventRecord(start, 0);
    MulMAdd2<float><<< blocks, threads >>>(gpu_mem, 10, 3.75, 0.355);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;
    double repeatF = 1.1e07 / (double)t;
    fprintf (stdout, "Adjust repeat factor = %lg\n", repeatF);

    delete[] hostMem;
    cudaFree((void*)gpu_mem);
    CHECK_CUDA_ERROR();

    // Initialize progress bar. We have 16 generic kernels and 2 hand tuned kernels.
    // Each kernel is executed 'passes' number of times for each single precision and
    // double precision (if avaialble).
    int totalRuns = 18*passes;
    if (doDouble)
       totalRuns <<= 1;  // multiply by 2
    ProgressBar pb(totalRuns);
    if (!verbose && !quiet)
       pb.Show(stdout);

    // Run single precision kernels
    RunTest<float> (resultDB, passes, verbose, quiet,
             repeatF, pb, "-SP");

    if (doDouble)
        RunTest<double> (resultDB, passes, verbose, quiet,
             repeatF, pb, "-DP");
    else
    {
        const char atts[] = "DP_Not_Supported";
        for (int pas=0 ; pas<passes ; ++pas)
        {
            resultDB.AddResult("Add1-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("Add2-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("Add4-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("Add8-DP", atts, "GFLOPS", FLT_MAX);

            resultDB.AddResult("Mul1-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("Mul2-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("Mul4-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("Mul8-DP", atts, "GFLOPS", FLT_MAX);

            resultDB.AddResult("MAdd1-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("MAdd2-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("MAdd4-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("MAdd8-DP", atts, "GFLOPS", FLT_MAX);

            resultDB.AddResult("MulMAdd1-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("MulMAdd2-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("MulMAdd4-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("MulMAdd8-DP", atts, "GFLOPS", FLT_MAX);

            // we deal with these separately
            //resultDB.AddResult("MulMAddU-DP", atts, "GFLOPS", FLT_MAX);
            //resultDB.AddResult("MAddU-DP", atts, "GFLOPS", FLT_MAX);
        }
    }

    // Problem Size
    int w = 2048, h = 2048;

    float root2 = 1.4142;
    if (repeatF<1)
       while (repeatF*root2<1) {
          repeatF*=2;
          if (w>h) w >>= 1;
          else  h >>= 1;
       }
/*
    When auto-scaling up, we must make sure that we do not exceed
    some device limit for block size. Disable for now.
 */
/*
    else
       while (repeatF>root2) {
          repeatF *= 0.5;
          if (w>h) h <<= 1;
          else  w <<= 1;
       }
*/
    const int nbytes_sp = w * h * sizeof(float);

    // Allocate gpu memory
    float *target_sp;
    cudaMalloc((void**)&target_sp, nbytes_sp);
    CHECK_CUDA_ERROR();

    // Get a couple non-zero random numbers
    float val1 = 0, val2 = 0;
    while (val1==0 || val2==0)
    {
        val1 = drand48();
        val2 = drand48();
    }

    blocks.x  = (w*h)/threads.x;
    for (int p = 0; p < passes; p++)
    {
        t = 0.0f;
        cudaEventRecord(start, 0);
        MAddU<<< blocks, threads >>>(target_sp, val1, val2);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        CHECK_CUDA_ERROR();
        cudaEventElapsedTime(&t, start, stop);
        t /= 1.e3;
        // Add result
        char atts[1024];
        long int nflopsPerPixel = ((2*32)*10*10*5) + 61;
        sprintf(atts, "Size:%d", w*h);
        resultDB.AddResult("MAddU-SP", atts, "GFLOPS",
                (((double)nflopsPerPixel)*w*h) / (t*1.e9));

        // update progress bar
        pb.addItersDone();
        if (!verbose && !quiet)
           pb.Show(stdout);

        cudaEventRecord(start, 0);
        MulMAddU<<< blocks, threads >>>(target_sp, val1, val2);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        CHECK_CUDA_ERROR();
        cudaEventElapsedTime(&t, start, stop);
        t /= 1.e3;

        // Add result
        nflopsPerPixel = ((3*8)*10*10*5) + 13;
        sprintf(atts, "Size:%d",w*h);
        resultDB.AddResult("MulMAddU-SP", atts, "GFLOPS",
                (((double)nflopsPerPixel)*w*h) / (t*1.e9));

        // update progress bar
        pb.addItersDone();
        if (!verbose && !quiet)
           pb.Show(stdout);
    }
    cudaFree((void*)target_sp);
    CHECK_CUDA_ERROR();

    if (doDouble)
    {
        const int nbytes_dp = w * h * sizeof(double);
        double *target_dp;
        cudaMalloc((void**)&target_dp, nbytes_dp);
        CHECK_CUDA_ERROR();

        // Thread block configuration
        dim3 threads(BLOCK_SIZE_DP,1,1);
        dim3 blocks((w*h)/BLOCK_SIZE_DP,1,1);

        const unsigned int passes = op.getOptionInt("passes");
        for (int p = 0; p < passes; p++)
        {

            cudaEventRecord(start, 0);
            MAddU_DP<<< blocks, threads >>>(target_dp, val1, val2);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            CHECK_CUDA_ERROR();
            cudaEventElapsedTime(&t, start, stop);
            t /= 1.e3;

            // Add result
            char atts[1024];
            long int nflopsPerPixel = ((2*32)*10*10*5) + 61;
            sprintf(atts, "Size:%d", w*h);
            resultDB.AddResult("MAddU-DP", atts, "GFLOPS",
                    (((double)nflopsPerPixel)*w*h) / (t*1.e9));

            // update progress bar
            pb.addItersDone();
            if (!verbose && !quiet)
               pb.Show(stdout);

            cudaEventRecord(start, 0);
            MulMAddU_DP<<< blocks, threads >>>(target_dp, val1, val2);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            CHECK_CUDA_ERROR();
            cudaEventElapsedTime(&t, start, stop);
            t /= 1.e3;

            // Add result
            nflopsPerPixel = ((3*8)*10*10*5) + 13;
            sprintf(atts, "Size:%d",w*h);
            resultDB.AddResult("MulMAddU-DP", atts, "GFLOPS",
                    (((double)nflopsPerPixel)*w*h) / (t*1.e9));

            // update progress bar
            pb.addItersDone();
            if (!verbose && !quiet)
               pb.Show(stdout);
        }
        cudaFree((void*)target_dp);
        CHECK_CUDA_ERROR();
    }
    else
    {
        // Add result
        char atts[1024];
        sprintf(atts, "Size:%d", w * h);
        // resultDB requires neg entry for every possible result
        const unsigned int passes = op.getOptionInt("passes");
        for (int p = 0; p < passes; p++) {
            resultDB.AddResult("MAddU-DP", atts, "GFLOPS", FLT_MAX);
            resultDB.AddResult("MulMAddU-DP", atts, "GFLOPS", FLT_MAX);
        }
    }

    if (!verbose)
        fprintf (stdout, "\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ****************************************************************************
// Function: RunTest
//
// Purpose:
//   Template function used for specializing the generic kernels for
//   single precision and double precision.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//
// Returns:  nothing
//
// Programmer: Gabriel Marin
// Creation: January 13, 2010
//
// ****************************************************************************
template <class T> void
RunTest(ResultDatabase &resultDB,
        int npasses,
        int verbose,
        int quiet,
        float repeatF,
        ProgressBar &pb,
        const char* precision)
{
    T *gpu_mem;
    char sizeStr[128];
    T *hostMem, *hostMem2;

    int realRepeats = (int)::round(repeatF*20);
    if (realRepeats < 2)
       realRepeats = 2;

    // Alloc host memory
    int halfNumFloats = 1024*1024;
    int numFloats = 2*halfNumFloats;
    hostMem = new T[numFloats];
    hostMem2 = new T[numFloats];

    cudaMalloc((void**)&gpu_mem, numFloats*sizeof(T));
    CHECK_CUDA_ERROR();

    // Variables used for timing
    float t = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CHECK_CUDA_ERROR();

    // Thread block configuration
    dim3 threads(128,1,1);
    dim3 blocks((numFloats)/128,1,1);

    for (int pass=0 ; pass<npasses ; ++pass)
    {
       // Benchmark each generic kernel. Generate new random numbers for each run.
       ////////// Add1 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the Add1 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       Add1<T><<< blocks, threads >>>(gpu_mem, realRepeats, 10.0);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       double flopCount = (double)numFloats * 1 * realRepeats * 240 * 1;
       double gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("Add1")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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


       ////////// Add2 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the Add2 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       Add2<T><<< blocks, threads >>>(gpu_mem, realRepeats, 10.0);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 1 * realRepeats * 120 * 2;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("Add2")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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

       ////////// Add4 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the Add4 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       Add4<T><<< blocks, threads >>>(gpu_mem, realRepeats, 10.0);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 1 * realRepeats * 60 * 4;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("Add4")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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

       ////////// Add8 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the Add8 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       Add8<T><<< blocks, threads >>>(gpu_mem, realRepeats, 10.0);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 1 * realRepeats * 30 * 8;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("Add8")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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


       ////////// Mul1 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the Mul1 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       Mul1<T><<< blocks, threads >>>(gpu_mem, realRepeats, 1.01);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 2 * realRepeats * 200 * 1;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("Mul1")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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


       ////////// Mul2 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the Mul2 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       Mul2<T><<< blocks, threads >>>(gpu_mem, realRepeats, 1.01);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 2 * realRepeats * 100 * 2;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("Mul2")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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

       ////////// Mul4 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the Mul4 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       Mul4<T><<< blocks, threads >>>(gpu_mem, realRepeats, 1.01);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 2 * realRepeats * 50 * 4;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("Mul4")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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

       ////////// Mul8 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the Mul8 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       Mul8<T><<< blocks, threads >>>(gpu_mem, realRepeats, 1.01);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 2 * realRepeats * 25 * 8;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("Mul8")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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


       ////////// MAdd1 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the MAdd1 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       MAdd1<T><<< blocks, threads >>>(gpu_mem, realRepeats, 10.0, 0.9899);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 2 * realRepeats * 240 * 1;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("MAdd1")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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


       ////////// MAdd2 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the MAdd2 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       MAdd2<T><<< blocks, threads >>>(gpu_mem, realRepeats, 10.0, 0.9899);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 2 * realRepeats * 120 * 2;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("MAdd2")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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

       ////////// MAdd4 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the MAdd4 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       MAdd4<T><<< blocks, threads >>>(gpu_mem, realRepeats, 10.0, 0.9899);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 2 * realRepeats * 60 * 4;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("MAdd4")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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

       ////////// MAdd8 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the MAdd8 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       MAdd8<T><<< blocks, threads >>>(gpu_mem, realRepeats, 10.0, 0.9899);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 2 * realRepeats * 30 * 8;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("MAdd8")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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


       ////////// MulMAdd1 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the MulMAdd1 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       MulMAdd1<T><<< blocks, threads >>>(gpu_mem, realRepeats, 3.75, 0.355);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 3 * realRepeats * 160 * 1;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("MulMAdd1")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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


       ////////// MulMAdd2 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the MulMAdd2 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       MulMAdd2<T><<< blocks, threads >>>(gpu_mem, realRepeats, 3.75, 0.355);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 3 * realRepeats * 80 * 2;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("MulMAdd2")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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

       ////////// MulMAdd4 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the MulMAdd4 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       MulMAdd4<T><<< blocks, threads >>>(gpu_mem, realRepeats, 3.75, 0.355);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 3 * realRepeats * 40 * 4;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("MulMAdd4")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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

       ////////// MulMAdd8 //////////
       // Initialize host data, with the first half the same as the second
       for (int j=0; j<halfNumFloats; ++j)
       {
           hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
       }

       // copy host memory to GPU memory
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(gpu_mem, hostMem, numFloats*sizeof(T), cudaMemcpyHostToDevice);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

       // Execute the MulMAdd8 kernel
       t = 0.0f;
       cudaEventRecord(start, 0);
       MulMAdd8<T><<< blocks, threads >>>(gpu_mem, realRepeats, 3.75, 0.355);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);
       CHECK_CUDA_ERROR();
       cudaEventElapsedTime(&t, start, stop);
       t *= 1.e6;

       // flopCount = numFloats(pixels) * flopCount/op * numLoopIters * unrollFactor * numStreams
       flopCount = (double)numFloats * 3 * realRepeats * 20 * 8;
       gflop = flopCount / (double)(t);

       sprintf (sizeStr, "Size:%07d", numFloats);
       resultDB.AddResult(string("MulMAdd8")+precision, sizeStr, "GFLOPS", gflop);

       // Zero out the test host memory
       for (int j=0 ; j<numFloats ; ++j)
           hostMem2[j] = 0.0;

       // Read the result device memory back to the host
       cudaEventRecord(start, 0); // do I even need this if I do not need the time?
       cudaMemcpy(hostMem2, gpu_mem, numFloats*sizeof(T), cudaMemcpyDeviceToHost);
       cudaEventRecord(stop, 0);
       cudaEventSynchronize(stop);

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

    delete[] hostMem;
    delete[] hostMem2;
    cudaFree((void*)gpu_mem);
    CHECK_CUDA_ERROR();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// Macros used to construct MaxFlops kernels
// Each mad OP is 32*2 = 64 FLOPS
#define OP {                                    \
        s0 = s6*s5 + s28;                       \
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
        s31 = s5*s4 + s27;                      \
    }

// so Each OP10 is 640 FLOPS
#define OP10 { OP OP OP OP OP OP OP OP OP OP }


// Each mad+mul MMOP is 8*3 = 24 FLOPS
#define MMOP {                                  \
        s0 = s4*s4 + s4;                        \
        s6 = s0*s5;                             \
        s1 = s5*s5 + s5;                        \
        s7 = s1*s6;                             \
        s2 = s6*s6 + s6;                        \
        s0 = s2*s7;                             \
        s3 = s7*s7 + s7;                        \
        s1 = s3*s0;                             \
        s4 = s0*s0 + s0;                        \
        s2 = s4*s1;                             \
        s5 = s1*s1 + s1;                        \
        s3 = s5*s2;                             \
        s6 = s2*s2 + s2;                        \
        s4 = s6*s3;                             \
        s7 = s3*s3 + s3;                        \
        s5 = s7*s4;                             \
    }

// So each OP10 is 240 FLOPS
#define MMOP10 { MMOP MMOP MMOP MMOP MMOP MMOP MMOP MMOP MMOP MMOP }



// Benchmark Kernels
__global__ void MAddU(float *target, float val1, float val2)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    // Create a bunch of local variables we can use up to 32 steps..
    register float v0=val1,     v1=val2,     v2=v0+v1,    v3=v0+v2;
    register float v4=v0+v3,    v5=v0+v4,    v6=v0+v5,    v7=v0+v6;
    register float v8=v0+v7,    v9=v0+v8,    v10=v0+v9,   v11=v0+v10;
    register float v12=v0+v11,  v13=v0+v12,  v14=v0+v13,  v15=v0+v14;
    register float v16=v0+v15,  v17=v16+v0,  v18=v16+v1,  v19=v16+v2;
    register float v20=v16+v3,  v21=v16+v4,  v22=v16+v5,  v23=v16+v6;
    register float v24=v16+v7,  v25=v16+v8,  v26=v16+v9,  v27=v16+v10;
    register float v28=v16+v11, v29=v16+v12, v30=v16+v13, v31=v16+v14;
    register float s0=v0,   s1=v1,   s2=v2,   s3=v3;
    register float s4=v4,   s5=v5,   s6=v6,   s7=v7;
    register float s8=v8,   s9=v9,   s10=v10, s11=v11;
    register float s12=v12, s13=v13, s14=v14, s15=v15;
    register float s16=v16, s17=v17, s18=v18, s19=v19;
    register float s20=v20, s21=v21, s22=v22, s23=v23;
    register float s24=v24, s25=v25, s26=v26, s27=v27;
    register float s28=v28, s29=v29, s30=v30, s31=v31;

    // 10 OP10s inside the loop = 6400 FLOPS in the .ptx code
    // and 5 loops of 10 OP10s = 32000 FLOPS per pixel total
    for (int i=0; i<5; i++)
    {
        OP10; OP10; OP10; OP10; OP10;
        OP10; OP10; OP10; OP10; OP10;
    }

    float result = (s0+s1+s2+s3+s4+s5+s6+s7+
                    s8+s9+s10+s11+s12+s13+s14+s15 +
                    s16+s17+s18+s19+s20+s21+s22+s23+
                    s24+s25+s26+s27+s28+s29+s30+s31);

    target[index] = result;
}

__global__ void MAddU_DP(double *target, double val1, double val2)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    register double v0=val1,     v1=val2,     v2=v0+v1,    v3=v0+v2;
    register double v4=v0+v3,    v5=v0+v4,    v6=v0+v5,    v7=v0+v6;
    register double v8=v0+v7,    v9=v0+v8,    v10=v0+v9,   v11=v0+v10;
    register double v12=v0+v11,  v13=v0+v12,  v14=v0+v13,  v15=v0+v14;
    register double v16=v0+v15,  v17=v16+v0,  v18=v16+v1,  v19=v16+v2;
    register double v20=v16+v3,  v21=v16+v4,  v22=v16+v5,  v23=v16+v6;
    register double v24=v16+v7,  v25=v16+v8,  v26=v16+v9,  v27=v16+v10;
    register double v28=v16+v11, v29=v16+v12, v30=v16+v13, v31=v16+v14;
    register double s0=v0,   s1=v1,   s2=v2,   s3=v3;
    register double s4=v4,   s5=v5,   s6=v6,   s7=v7;
    register double s8=v8,   s9=v9,   s10=v10, s11=v11;
    register double s12=v12, s13=v13, s14=v14, s15=v15;
    register double s16=v16, s17=v17, s18=v18, s19=v19;
    register double s20=v20, s21=v21, s22=v22, s23=v23;
    register double s24=v24, s25=v25, s26=v26, s27=v27;
    register double s28=v28, s29=v29, s30=v30, s31=v31;


    // 10 OP10s inside the loop = 6400 FLOPS in the .ptx code
    // and 5 loops of 10 OP10s = 32000 FLOPS per pixel total
    for (int i=0; i<5; i++)
    {
        OP10; OP10; OP10; OP10; OP10;
        OP10; OP10; OP10; OP10; OP10;
    }
    double result = (s0+s1+s2+s3+s4+s5+s6+s7+
                    s8+s9+s10+s11+s12+s13+s14+s15 +
                    s16+s17+s18+s19+s20+s21+s22+s23+
                    s24+s25+s26+s27+s28+s29+s30+s31);
    target[index] = result;
}


__global__ void MulMAddU(float *target, float val1, float val2)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    register float v0=val1,     v1=val2,     v2=v0+v1,    v3=v0+v2;
    register float v4=v0+v3,    v5=v0+v4,    v6=v0+v5,    v7=v0+v6;
    register float v8=v0+v7,    v9=v0+v8,    v10=v0+v9,   v11=v0+v10;
    register float v12=v0+v11,  v13=v0+v12,  v14=v0+v13,  v15=v0+v14;
    register float v16=v0+v15,  v17=v16+v0,  v18=v16+v1,  v19=v16+v2;
    register float v20=v16+v3,  v21=v16+v4,  v22=v16+v5,  v23=v16+v6;
    register float v24=v16+v7,  v25=v16+v8,  v26=v16+v9,  v27=v16+v10;
    register float v28=v16+v11, v29=v16+v12, v30=v16+v13, v31=v16+v14;
    register float s0=v0,   s1=v1,   s2=v2,   s3=v3;
    register float s4=v4,   s5=v5,   s6=v6,   s7=v7;
    register float s8=v8,   s9=v9,   s10=v10, s11=v11;
    register float s12=v12, s13=v13, s14=v14, s15=v15;
    register float s16=v16, s17=v17, s18=v18, s19=v19;
    register float s20=v20, s21=v21, s22=v22, s23=v23;
    register float s24=v24, s25=v25, s26=v26, s27=v27;
    register float s28=v28, s29=v29, s30=v30, s31=v31;

    // 10 OP10s inside the loop = 2400 FLOPS in the .ptx code
    // and 5 loops of 10 OP10s = 12000 FLOPS per pixel total
    for (int i=0; i<5; i++)
    {
        MMOP10; MMOP10; MMOP10; MMOP10; MMOP10;
        MMOP10; MMOP10; MMOP10; MMOP10; MMOP10;
    }
    float result = (s0+s1+s2+s3+s4+s5+s6+s7+
                    s8+s9+s10+s11+s12+s13+s14+s15 +
                    s16+s17+s18+s19+s20+s21+s22+s23+
                    s24+s25+s26+s27+s28+s29+s30+s31);

    target[index] = result;
}

__global__ void MulMAddU_DP(double *target, double val1, double val2)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    register double v0=val1,     v1=val2,     v2=v0+v1,    v3=v0+v2;
    register double v4=v0+v3,    v5=v0+v4,    v6=v0+v5,    v7=v0+v6;
    register double v8=v0+v7,    v9=v0+v8,    v10=v0+v9,   v11=v0+v10;
    register double v12=v0+v11,  v13=v0+v12,  v14=v0+v13,  v15=v0+v14;
    register double v16=v0+v15,  v17=v16+v0,  v18=v16+v1,  v19=v16+v2;
    register double v20=v16+v3,  v21=v16+v4,  v22=v16+v5,  v23=v16+v6;
    register double v24=v16+v7,  v25=v16+v8,  v26=v16+v9,  v27=v16+v10;
    register double v28=v16+v11, v29=v16+v12, v30=v16+v13, v31=v16+v14;
    register double s0=v0,   s1=v1,   s2=v2,   s3=v3;
    register double s4=v4,   s5=v5,   s6=v6,   s7=v7;
    register double s8=v8,   s9=v9,   s10=v10, s11=v11;
    register double s12=v12, s13=v13, s14=v14, s15=v15;
    register double s16=v16, s17=v17, s18=v18, s19=v19;
    register double s20=v20, s21=v21, s22=v22, s23=v23;
    register double s24=v24, s25=v25, s26=v26, s27=v27;
    register double s28=v28, s29=v29, s30=v30, s31=v31;

    // 10 OP10s inside the loop = 2400 FLOPS in the .ptx code
    // and 5 loops of 10 OP10s = 12000 FLOPS per pixel total
    for (int i=0; i<5; i++)
    {
        MMOP10; MMOP10; MMOP10; MMOP10; MMOP10;
        MMOP10; MMOP10; MMOP10; MMOP10; MMOP10;
    }
    double result = (s0+s1+s2+s3+s4+s5+s6+s7+
                    s8+s9+s10+s11+s12+s13+s14+s15 +
                    s16+s17+s18+s19+s20+s21+s22+s23+
                    s24+s25+s26+s27+s28+s29+s30+s31);
    target[index] = result;
}

// v = 10.0
#define ADD1_OP   s=v-s;
#define ADD2_OP   ADD1_OP s2=v-s2;
#define ADD4_OP   ADD2_OP s3=v-s3; s4=v-s4;
#define ADD8_OP   ADD4_OP s5=v-s5; s6=v-s6; s7=v-s7; s8=v-s8;

// v = 1.01
#define MUL1_OP   s=s*s*v;
#define MUL2_OP   MUL1_OP s2=s2*s2*v;
#define MUL4_OP   MUL2_OP s3=s3*s3*v; s4=s4*s4*v;
#define MUL8_OP   MUL4_OP s5=s5*s5*v; s6=s6*s6*v; s7=s7*s7*v; s8=s8*s8*v;

// v1 = 10.0, v2 = 0.9899
#define MADD1_OP  s=v1-s*v2;
#define MADD2_OP  MADD1_OP s2=v1-s2*v2;
#define MADD4_OP  MADD2_OP s3=v1-s3*v2; s4=v1-s4*v2;
#define MADD8_OP  MADD4_OP s5=v1-s5*v2; s6=v1-s6*v2; s7=v1-s7*v2; s8=v1-s8*v2;

// v1 = 3.75, v2 = 0.355
#define MULMADD1_OP  s=(v1-v2*s)*s;
#define MULMADD2_OP  MULMADD1_OP s2=(v1-v2*s2)*s2;
#define MULMADD4_OP  MULMADD2_OP s3=(v1-v2*s3)*s3; s4=(v1-v2*s4)*s4;
#define MULMADD8_OP  MULMADD4_OP s5=(v1-v2*s5)*s5; s6=(v1-v2*s6)*s6; s7=(v1-v2*s7)*s7; s8=(v1-v2*s8)*s8;

#define ADD1_MOP20  \
     ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP \
     ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP
#define ADD2_MOP20  \
     ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP \
     ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP
#define ADD4_MOP10  \
     ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP \
     ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP
#define ADD8_MOP5  \
     ADD8_OP ADD8_OP ADD8_OP ADD8_OP ADD8_OP

#define MUL1_MOP20  \
     MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP \
     MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP
#define MUL2_MOP20  \
     MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP \
     MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP
#define MUL4_MOP10  \
     MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP \
     MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP
#define MUL8_MOP5  \
     MUL8_OP MUL8_OP MUL8_OP MUL8_OP MUL8_OP

#define MADD1_MOP20  \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP
#define MADD2_MOP20  \
     MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP \
     MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP
#define MADD4_MOP10  \
     MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP \
     MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP
#define MADD8_MOP5  \
     MADD8_OP MADD8_OP MADD8_OP MADD8_OP MADD8_OP

#define MULMADD1_MOP20  \
     MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP \
     MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP
#define MULMADD2_MOP20  \
     MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP \
     MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP
#define MULMADD4_MOP10  \
     MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP \
     MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP
#define MULMADD8_MOP5  \
     MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP


template <class T>
__global__ void Add1(T *data, int nIters, T v) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid];
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 20 operations.
        Unroll 12 more times for 240 operations total.
      */
     ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20
     ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20
  }
  data[gid] = s;
}

template <class T>
__global__ void Add2(T *data, int nIters, T v) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid], s2=10.0f-s;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 20 operations.
        Unroll 6 more times for 120 operations total.
      */
     ADD2_MOP20 ADD2_MOP20 ADD2_MOP20
     ADD2_MOP20 ADD2_MOP20 ADD2_MOP20
  }
  data[gid] = s+s2;
}

template <class T>
__global__ void Add4(T *data, int nIters, T v) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid], s2=10.0f-s, s3=9.0f-s, s4=9.0f-s2;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 10 operations.
        Unroll 6 more times for 60 operations total.
      */
     ADD4_MOP10 ADD4_MOP10 ADD4_MOP10
     ADD4_MOP10 ADD4_MOP10 ADD4_MOP10
  }
  data[gid] = (s+s2)+(s3+s4);
}

template <class T>
__global__ void Add8(T *data, int nIters, T v) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid], s2=10.0f-s, s3=9.0f-s, s4=9.0f-s2, s5=8.0f-s, s6=8.0f-s2, s7=7.0f-s, s8=7.0f-s2;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 5 operations.
        Unroll 6 more times for 30 operations total.
      */
     ADD8_MOP5 ADD8_MOP5 ADD8_MOP5
     ADD8_MOP5 ADD8_MOP5 ADD8_MOP5
  }
  data[gid] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
}


template <class T>
__global__ void Mul1(T *data, int nIters, T v) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid]-data[gid]+0.999f;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 20 operations.
        Unroll 10 more times for 200 operations total.
      */
     MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
     MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
  }
  data[gid] = s;
}

template <class T>
__global__ void Mul2(T *data, int nIters, T v) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid]-data[gid]+0.999f, s2=s-0.0001f;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 20 operations.
        Unroll 5 more times for 100 operations total.
      */
     MUL2_MOP20 MUL2_MOP20 MUL2_MOP20
     MUL2_MOP20 MUL2_MOP20
  }
  data[gid] = s+s2;
}

template <class T>
__global__ void Mul4(T *data, int nIters, T v) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid]-data[gid]+0.999f, s2=s-0.0001f, s3=s-0.0002f, s4=s-0.0003f;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 10 operations.
        Unroll 5 more times for 50 operations total.
      */
     MUL4_MOP10 MUL4_MOP10 MUL4_MOP10
     MUL4_MOP10 MUL4_MOP10
  }
  data[gid] = (s+s2)+(s3+s4);
}

template <class T>
__global__ void Mul8(T *data, int nIters, T v) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid]-data[gid]+0.999f, s2=s-0.0001f, s3=s-0.0002f, s4=s-0.0003f, s5=s-0.0004f, s6=s-0.0005f, s7=s-0.0006f, s8=s-0.0007f;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 5 operations.
        Unroll 5 more times for 25 operations total.
      */
     MUL8_MOP5 MUL8_MOP5 MUL8_MOP5
     MUL8_MOP5 MUL8_MOP5
  }
  data[gid] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
}


template <class T>
__global__ void MAdd1(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid];
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 20 operations.
        Unroll 12 more times for 240 operations total.
      */
     MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
     MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
  }
  data[gid] = s;
}

template <class T>
__global__ void MAdd2(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid], s2=10.0f-s;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 20 operations.
        Unroll 6 more times for 120 operations total.
      */
     MADD2_MOP20 MADD2_MOP20 MADD2_MOP20
     MADD2_MOP20 MADD2_MOP20 MADD2_MOP20
  }
  data[gid] = s+s2;
}

template <class T>
__global__ void MAdd4(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid], s2=10.0f-s, s3=9.0f-s, s4=9.0f-s2;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 10 operations.
        Unroll 6 more times for 60 operations total.
      */
     MADD4_MOP10 MADD4_MOP10 MADD4_MOP10
     MADD4_MOP10 MADD4_MOP10 MADD4_MOP10
  }
  data[gid] = (s+s2)+(s3+s4);
}

template <class T>
__global__ void MAdd8(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid], s2=10.0f-s, s3=9.0f-s, s4=9.0f-s2, s5=8.0f-s, s6=8.0f-s2, s7=7.0f-s, s8=7.0f-s2;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 5 operations.
        Unroll 6 more times for 30 operations total.
      */
     MADD8_MOP5 MADD8_MOP5 MADD8_MOP5
     MADD8_MOP5 MADD8_MOP5 MADD8_MOP5
  }
  data[gid] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
}


template <class T>
__global__ void MulMAdd1(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid];
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 20 operations.
        Unroll 8 more times for 160 operations total.
      */
     MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
     MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
  }
  data[gid] = s;
}

template <class T>
__global__ void MulMAdd2(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid], s2=10.0f-s;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 20 operations.
        Unroll 4 more times for 80 operations total.
      */
     MULMADD2_MOP20 MULMADD2_MOP20
     MULMADD2_MOP20 MULMADD2_MOP20
  }
  data[gid] = s+s2;
}

template <class T>
__global__ void MulMAdd4(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid], s2=10.0f-s, s3=9.0f-s, s4=9.0f-s2;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 10 operations.
        Unroll 4 more times for 40 operations total.
      */
     MULMADD4_MOP10 MULMADD4_MOP10
     MULMADD4_MOP10 MULMADD4_MOP10
  }
  data[gid] = (s+s2)+(s3+s4);
}

template <class T>
__global__ void MulMAdd8(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  register T s = data[gid], s2=10.0f-s, s3=9.0f-s, s4=9.0f-s2, s5=8.0f-s, s6=8.0f-s2, s7=7.0f-s, s8=7.0f-s2;
  for (int j=0 ; j<nIters ; ++j) {
     /* Each macro op has 5 operations.
        Unroll 4 more times for 20 operations total.
      */
     MULMADD8_MOP5 MULMADD8_MOP5
     MULMADD8_MOP5 MULMADD8_MOP5
  }
  data[gid] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
}

