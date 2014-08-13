// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Jeffrey Vetter <vetter@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
// Copyright (c) 2011-2013, UT-Battelle, LLC
// Copyright (c) 2013, Intel Corporation
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//   
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Oak Ridge National Laboratory, nor UT-Battelle, LLC, 
//    nor the names of its contributors may be used to endorse or promote 
//    products derived from this software without specific prior written 
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
// THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include <string>
#include <stdio.h>
#include "omp.h"
#include "Timer.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

//Based on the 5110P with 4 threads per core
#define MIC_THREADS 240
// For heterogeneous features include "offload.h"
#include "offload.h"
#ifdef __MIC__ ||__MIC2__
#include <immintrin.h>
#endif

// Memory Benchmarks Sizes
#define VECSIZE_SP 480000
#define REPS_SP 1000

float __declspec(target(mic)) testICC_read(const int reps);
float __declspec(target(mic)) testICC_write(const int reps, const float value);


// L2 & L1 Benchmarks Sizes
#define VECSIZE_SP_L2 4864
#define REPS_SP_L2 1000000
#define VECSIZE_SP_L1 1024
#define REPS_SP_L1 1000000

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
// Programmer: Alexander Heinecke
// Creation: July 23, 2010
//
// Modifications:
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    // No specific options for this benchmark.
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Alexander Heinecke
// Creation: July 23, 2010
//
// Modifications:
// Dec. 12, 2012 - Kyle Spafford - Updates and SHOC coding style conformance.
// Aug. 12, 2014 - Jeff Young - Removed intrinsics code and eversion variable 
//
// ****************************************************************************
void RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
    const bool verbose = op.getOptionBool("verbose");
    const unsigned int passes = op.getOptionInt("passes");

    char sizeStr[128];

    double t = 0.0f;
    double startTime;
    unsigned int w;
    __declspec(target(mic)) static unsigned int reps;
    double nbytes;
    float res = 0.0;
    float input = 1.0;

    int numThreads = MIC_THREADS;
    double dThreads = static_cast<double>(numThreads);
    
    double bdwth;

    for (int p = 0; p < passes; p++)
    {
        cout << "Running benchmarks, pass: " << p << "\n";

        // Test Memory
        w = VECSIZE_SP;
        reps = REPS_SP;

        // ========= Test Read - ICC Code =============
        int testICC_readTimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testICC_read(reps);
        t = Timer::Stop(testICC_readTimerHandle, "testICC_read");

        // Add Result - while this is not strictly a coalesced read, this value matches up with the gmem_writebw result for SHOC
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*dThreads;
        bdwth = ((double)nbytes) / (t*1.e9);
        resultDB.AddResult("readGlobalMemoryCoalesced", sizeStr, "GB/s",
                bdwth);

        // ========= Test Write - ICC Code =============
        int testICC_writeTimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testICC_write(reps, input);
        t = Timer::Stop(testICC_writeTimerHandle, "testICC_write");

        // Add Result - while this is not strictly a coalesced write, this value matches up with the gmem_writebw result for SHOC
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*dThreads;
        bdwth = ((double)nbytes) / (t*1.e9);
        resultDB.AddResult("writeGlobalMemoryCoalesced", sizeStr, "GB/s",
                bdwth);
    }
}
// ****************************************************************************
// Function: testICC_read
//
// Purpose: RUns the 
//
// Arguments:
//
// Returns:  nothing
//
// Programmer: Alexander Heinecke
// Creation: July 23, 2010
//
// Modifications:
// Aug. 12, 2014 - Jeff Young - Removed eversion variable
//
// ****************************************************************************


float __declspec(target(mic)) testICC_read(const int reps)
{
#ifdef __MIC__ || __MIC2__

    size_t numElements;

    numElements = VECSIZE_SP*MIC_THREADS;

    float* a = (float*)_mm_malloc(sizeof(float)*numElements, 64);
    __declspec(aligned(64))float res = 0.0;
    #pragma ivdep
    #pragma omp parallel for shared(a)
    for (int q = 0; q < numElements; q++)
    {
        a[q] = 1.0;
    }

    #pragma omp parallel shared(res)
    {
        __declspec(aligned(64))float b = 0.0;
        int offset = VECSIZE_SP * omp_get_thread_num();

        for (int m = 0; m < reps; m++)
        {
            #pragma vector aligned
            #pragma ivdep
            for (int q = offset; q < offset+VECSIZE_SP; q++)
            {
                b += a[q];
            }
            b += 1.0;
        }
        #pragma omp critical
        {
            res += b;
        }
    }
    _mm_free(a);
    return res;
#else
    return 0.0;
#endif
}


float __declspec(target(mic)) testICC_write(const int reps, const float value)
{
#ifdef __MIC__ || __MIC2__

    size_t numElements;

    numElements = VECSIZE_SP*MIC_THREADS;

    float* a = (float*)_mm_malloc(sizeof(float)*numElements, 64);
    __declspec(aligned(64))float res = 0.0;

    #pragma vector aligned
    #pragma ivdep
    for (int q = 0; q < numElements; q++)
    {
        a[q] = 1.0;
    }

    #pragma omp parallel shared(res)
    {
        int offset = VECSIZE_SP * omp_get_thread_num();
        __declspec(aligned(64))float writeData = value + 
            static_cast<float>(omp_get_thread_num());

        for (int m = 0; m < reps; m++)
        {
            #pragma vector aligned
            #pragma ivdep
            for (int q = offset; q < offset+VECSIZE_SP; q++)
            {
                a[q] += writeData;
            }
            writeData += 1.0;
        }
    }

    // Sum something in a, avoid compiler optimizations
    res = a[0] + a[numElements-1];
    _mm_free(a);
    return res;
#else
    return 0.0;
#endif
}
