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

#define MIC_THREADS 240
// For heterogeneous features include "offload.h"
#include "offload.h"
#ifdef __MIC__ ||__MIC2__
#include <immintrin.h>
#endif

// Memory Benchmarks Sizes
#define VECSIZE_SP 480000
#define REPS_SP 1000

float __declspec(target(mic)) testICC_read(const int reps, const int eversion);
float __declspec(target(mic)) testICC_write(const int reps, 
        const int eversion, 
        const float value);

float __declspec(target(mic)) testIntrinsics_read(const int reps, 
        const int eversion);
float __declspec(target(mic)) testIntrinsics_write(const int reps, 
        const int eversion, 
        const float value);

// L2 & L1 Benchmarks Sizes
#define VECSIZE_SP_L2 4864
#define REPS_SP_L2 1000000
#define VECSIZE_SP_L1 1024
#define REPS_SP_L1 1000000

float __declspec(target(mic)) testICC_read_caches(const int reps, 
        const int eversion, 
        const int worksize);
float __declspec(target(mic)) testICC_write_caches(const int reps, 
        const int eversion, 
        const float value, 
        const int worksize);

float __declspec(target(mic)) testIntrinsics_read_caches(const int reps, 
        const int eversion, 
        const int worksize);
float __declspec(target(mic)) testIntrinsics_write_caches(const int reps, 
        const int eversion, 
        const float value, 
        const int worksize);

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
//
// ****************************************************************************
void RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
    const bool verbose = op.getOptionBool("verbose");
    const unsigned int passes = op.getOptionInt("passes");

    __declspec(target(mic)) static int eversion = 1;

    double t = 0.0f;
    double startTime;
    unsigned int w;
    __declspec(target(mic)) static unsigned int reps;
    double nbytes;
    float res = 0.0;
    float input = 1.0;

    int numThreads = MIC_THREADS;
    double dThreads = static_cast<double>(numThreads);

    for (int p = 0; p < passes; p++)
    {
        // Test Memory
        w = VECSIZE_SP;
        reps = REPS_SP;

        // Test Read - ICC Code
        int testICC_readTimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testICC_read(reps, eversion);
        t = Timer::Stop(testICC_readTimerHandle, "testICC_read");

        // Add result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*dThreads;
        resultDB.AddResult("READ_MEM", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));

// Remove intrinsic versions
#if 0
        // Test Read - Intrinsics Code
        int testIntrinsics_readTimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testIntrinsics_read(reps, eversion);
        t = Timer::Stop(testIntrinsics_readTimerHandle, "testIntrinsics_read");

        // Add result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*dThreads;
  resultDB.AddResult("INT_R_MEM", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));
#endif
        // Test Write - ICC Code
        int testICC_writeTimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testICC_write(reps, eversion, input);
        t = Timer::Stop(testICC_writeTimerHandle, "testICC_write");

        // Add Result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*dThreads;
        resultDB.AddResult("WRITE_MEM", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));
//Remove intrinsics version
#if 0
        // Test Write - Intrinsics Code
        int testIntrinsics_writeTimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testIntrinsics_write(reps, eversion, input);
        t = Timer::Stop(testIntrinsics_writeTimerHandle, "testIntrinsics_write");

        // Add result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*dThreads;
        resultDB.AddResult("INT_W_MEM", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));
#endif
// commnented out all the cache measurements as the code is buggy
#if 0
        // Begin L1 Tests
        w = VECSIZE_SP_L1;
        reps = REPS_SP_L1;

        // Test Read L1 - ICC Code
        int testICC_read_caches_l1TimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testICC_read_caches(reps, eversion, w);
        t = Timer::Stop(testICC_read_caches_l1TimerHandle, "testICC_read_caches L1");

        // Add result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*4.0;
        resultDB.AddResult("ICC_R_L1", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));

        // Test Read L1 - Intrinsics Code
        int testIntrinsics_read_caches_l1TimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testIntrinsics_read_caches(reps, eversion, w);
        t = Timer::Stop(testIntrinsics_read_caches_l1TimerHandle, "testIntrinsics_read_caches L1");

        // Add result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*4.0;
        resultDB.AddResult("INT_R_L1", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));

        // Test Write L1 - ICC Code
        int testICC_write_caches_l1TimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testICC_write_caches(reps, eversion, input, w);
        t = Timer::Stop(testICC_write_caches_l1TimerHandle, "testICC_write_caches L1");

        // Add result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*4.0;
        resultDB.AddResult("ICC_W_L1", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));

        // Test Write L1 - Intrinsics Code
        int testIntrinsics_write_caches_l1TimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testIntrinsics_write_caches(reps, eversion, input, w);
        t = Timer::Stop(testIntrinsics_write_caches_l1TimerHandle, "testInstrincs_write_caches L1");

        // Add result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*4.0;
        resultDB.AddResult("INT_W_L1", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));

        // Begin L2 Tests
        w = VECSIZE_SP_L2;
        reps = REPS_SP_L2;

        // Test Read L2 - ICC Code
        int testICC_read_caches_l2TimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testICC_read_caches(reps, eversion, w);
        t = Timer::Stop(testICC_read_caches_l2TimerHandle, "testICC_read_caches L2");

        // Add result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*4.0;
        resultDB.AddResult("ICC_R_L2", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));

        // Test Read L2 - Intrinsics Code
        int testIntrinsics_read_caches_l2TimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testIntrinsics_read_caches(reps, eversion, w);
        t = Timer::Stop(testIntrinsics_read_caches_l2TimerHandle, "testIntrinsics_read_caches L2");

        // Add result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*4.0;
        resultDB.AddResult("INT_R_L2", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));

        // Test Write L2 - ICC Code
        int testICC_write_caches_l2TimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testICC_write_caches(reps, eversion, input, w);
        t = Timer::Stop(testICC_write_caches_l2TimerHandle, "testICC_write_caches L2");

        // Add result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*4.0;
        resultDB.AddResult("ICC_W_L2", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));

        // Test Write L2 - Intrinsics Code
        int testIntrinsics_write_caches_l2TimerHandle = Timer::Start();
        #pragma offload target (mic)
        res = testIntrinsics_write_caches(reps, eversion, input, w);
        t = Timer::Stop(testIntrinsics_write_caches_l2TimerHandle, "testIntrinsics_write_caches L2");

        // Add result
        nbytes = ((double)w)*((double)reps)*((double)sizeof(float))*4.0;
        resultDB.AddResult("INT_W_L2", "", "GB/s",
                (((double)nbytes) / (t*1.e9)));
#endif
    }
}

float __declspec(target(mic)) testICC_read(const int reps, const int eversion)
{
#ifdef __MIC__ || __MIC2__

    size_t numElements;

    if (eversion == 1)
    {
        numElements = VECSIZE_SP*MIC_THREADS;
    }
    else if (eversion == 2)
    {
        numElements = VECSIZE_SP*108;
    }
    else
    {
        numElements = VECSIZE_SP*92;
    }

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

float __declspec(target(mic)) testIntrinsics_read(const int reps, 
        const int eversion)
{
#ifdef __MIC__ || __MIC2__

    size_t numElements;

    if (eversion == 1)
    {
        numElements = VECSIZE_SP*MIC_THREADS;
    }
    else if (eversion == 2)
    {
        numElements = VECSIZE_SP*108;
    }
    else
    {
        numElements = VECSIZE_SP*92;
    }

    float* a = (float*)_mm_malloc(sizeof(float)*numElements, 64);
    __declspec(aligned(64))float res = 0.0;

    #pragma vector aligned
    #pragma ivdep
    #pragma omp parallel for shared(a)
    for (int q = 0; q < numElements; q++)
    {
        a[q] = 1.0;
    }

    #pragma omp parallel shared(res)
    {
        int offset = VECSIZE_SP * omp_get_thread_num();
        __m512 b_0;
        __m512 b_1;
        __m512 b_2;
        __m512 b_3;
        __m512 b_4;
        __m512 b_5;
        __m512 b_6;
        __m512 b_7;

        for (int m = 0; m < reps; m++)
        {
            b_0 = _mm512_setzero();
            b_1 = _mm512_setzero();
            b_2 = _mm512_setzero();
            b_3 = _mm512_setzero();
            b_4 = _mm512_setzero();
            b_5 = _mm512_setzero();
            b_6 = _mm512_setzero();
            b_7 = _mm512_setzero();

            #pragma vector aligned
            #pragma ivdep
            for (int q = offset; q < offset+VECSIZE_SP; q+=128)
            {
                _mm_prefetch((const char *)&(a[q+128]), _MM_HINT_T0);
                _mm_prefetch((const char *)&(a[q+144]), _MM_HINT_T0);
                _mm_prefetch((const char *)&(a[q+160]), _MM_HINT_T0);
                _mm_prefetch((const char *)&(a[q+176]), _MM_HINT_T0);
                _mm_prefetch((const char *)&(a[q+192]), _MM_HINT_T0);
                _mm_prefetch((const char *)&(a[q+208]), _MM_HINT_T0);
                _mm_prefetch((const char *)&(a[q+224]), _MM_HINT_T0);
                _mm_prefetch((const char *)&(a[q+240]), _MM_HINT_T0);

                // For KNF, cheaply emulated to KNC
                __m512 a_0 = _mm512_load_ps(&(a[q]));
                __m512 a_1 = _mm512_load_ps(&(a[q+16]));
                __m512 a_2 = _mm512_load_ps(&(a[q+32]));
                __m512 a_3 = _mm512_load_ps(&(a[q+48]));
                __m512 a_4 = _mm512_load_ps(&(a[q+64]));
                __m512 a_5 = _mm512_load_ps(&(a[q+80]));
                __m512 a_6 = _mm512_load_ps(&(a[q+96]));
                __m512 a_7 = _mm512_load_ps(&(a[q+112]));

                b_0 = _mm512_add_ps(b_0, a_0);
                b_1 = _mm512_add_ps(b_1, a_1);
                b_2 = _mm512_add_ps(b_2, a_2);
                b_3 = _mm512_add_ps(b_3, a_3);
                b_4 = _mm512_add_ps(b_4, a_4);
                b_5 = _mm512_add_ps(b_5, a_5);
                b_6 = _mm512_add_ps(b_6, a_6);
                b_7 = _mm512_add_ps(b_7, a_7);
            }
            b_0 = _mm512_add_ps(b_0, b_1);
            b_2 = _mm512_add_ps(b_2, b_3);
            b_4 = _mm512_add_ps(b_4, b_5);
            b_6 = _mm512_add_ps(b_6, b_7);
            b_0 = _mm512_add_ps(b_0, b_2);
            b_4 = _mm512_add_ps(b_4, b_6);
            b_0 = _mm512_add_ps(b_0, b_4);
        }

        #pragma omp critical
        {
            res += _mm512_reduce_add_ps(b_0);
        }
    }
    _mm_free(a);
    return res;
#else
    return 0.0;
#endif
}

float __declspec(target(mic)) testICC_write(const int reps, 
        const int eversion, const float value)
{
#ifdef __MIC__ || __MIC2__

    size_t numElements;

    if (eversion == 1)
    {
        numElements = VECSIZE_SP*MIC_THREADS;
    }
    else if (eversion == 2)
    {
        numElements = VECSIZE_SP*108;
    }
    else
    {
        numElements = VECSIZE_SP*92;
    }

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

float __declspec(target(mic)) testIntrinsics_write(const int reps, 
        const int eversion, const float value)
{
#ifdef __MIC__ ||  __MIC2__

    size_t numElements;

    if (eversion == 1)
    {
        numElements = VECSIZE_SP*MIC_THREADS;
    }
    else if (eversion == 2)
    {
        numElements = VECSIZE_SP*108;
    }
    else
    {
        numElements = VECSIZE_SP*92;
    }

    float* a = (float*)_mm_malloc(sizeof(float)*numElements, 64);
    __declspec(aligned(64))float res = 0.0;

    #pragma omp parallel shared(res)
    {
        int offset = VECSIZE_SP * omp_get_thread_num();
        __declspec(aligned(64))float writeData = value + 
            static_cast<float>(omp_get_thread_num());

        __m512 toWrite = _mm512_load_ps(&(writeData));
        __m512 one = _mm512_set_1to16_ps(1.0);

        for (int m = 0; m < reps; m++)
        {
            #pragma vector aligned
            #pragma ivdep
            for (int q = offset; q < offset+VECSIZE_SP; q+=128)
            {
                // Only KNF according to spreadsheet
                _mm_prefetch((const char *)&(a[q+128]), _MM_HINT_ENTA);
                _mm_prefetch((const char *)&(a[q+144]), _MM_HINT_ENTA);
                _mm_prefetch((const char *)&(a[q+160]), _MM_HINT_ENTA);
                _mm_prefetch((const char *)&(a[q+176]), _MM_HINT_ENTA);
                _mm_prefetch((const char *)&(a[q+192]), _MM_HINT_ENTA);
                _mm_prefetch((const char *)&(a[q+208]), _MM_HINT_ENTA);
                _mm_prefetch((const char *)&(a[q+224]), _MM_HINT_ENTA);
                _mm_prefetch((const char *)&(a[q+240]), _MM_HINT_ENTA);

                __m512 a_0 = toWrite;
                __m512 a_1 = toWrite;
                __m512 a_2 = toWrite;
                __m512 a_3 = toWrite;
                __m512 a_4 = toWrite;
                __m512 a_5 = toWrite;
                __m512 a_6 = toWrite;
                __m512 a_7 = toWrite;

                // For KNF, cheaply converted to KNC
                _mm512_store_ps(&(a[q]), a_0);
                _mm512_store_ps(&(a[q+16]), a_1);
                _mm512_store_ps(&(a[q+32]), a_2);
                _mm512_store_ps(&(a[q+48]), a_3);
                _mm512_store_ps(&(a[q+64]), a_4);
                _mm512_store_ps(&(a[q+80]), a_5);
                _mm512_store_ps(&(a[q+96]), a_6);
                _mm512_store_ps(&(a[q+112]), a_7);
            }
            toWrite = _mm512_add_ps(toWrite, one);
        }
    }

    // Sum something in a to avoid compiler optimizations
    res = a[0] + a[numElements-1];
    _mm_free(a);
    return res;
#else
    return 0.0;
#endif
}

float __declspec(target(mic)) testICC_read_caches(const int reps, 
        const int eversion, const int worksize)
{
#ifdef __MIC__ ||  __MIC2__

    size_t numElements;
    numElements = worksize*4;

    float* a = (float*)_mm_malloc(sizeof(float)*numElements, 64);
    __declspec(aligned(64))float res = 0.0;

    #pragma vector aligned
    #pragma ivdep
    for (int q = 0; q < numElements; q++)
    {
        a[q] = 1.0;
    }

    #pragma omp parallel num_threads(4) shared(res)
    {
        __declspec(aligned(64))float b = 0.0;
        int offset = worksize * omp_get_thread_num();

        for (int m = 0; m < reps; m++)
        {
            #pragma vector aligned
            #pragma ivdep
            for (int q = offset; q < offset+worksize; q++)
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

float __declspec(target(mic)) testICC_write_caches(const int reps, 
        const int eversion, const float value, const int worksize)
{
#ifdef __MIC__ ||  __MIC2__
    size_t numElements;
    numElements = worksize*4;

    float* a = (float*)_mm_malloc(sizeof(float)*numElements, 64);
    __declspec(aligned(64))float res = 0.0;

    #pragma omp parallel num_threads(4) shared(res)
    {
        int offset = worksize * omp_get_thread_num();
        __declspec(aligned(64))float writeData = value + 
            static_cast<float>(omp_get_thread_num());

        for (int m = 0; m < reps; m++)
        {
            #pragma vector aligned
            #pragma ivdep
            for (int q = offset; q < offset+worksize; q++)
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

float __declspec(target(mic)) testIntrinsics_read_caches(const int reps, 
        const int eversion, const int worksize)
{
#ifdef __MIC__ || __MIC2__

    size_t numElements;
    numElements = worksize*4;

    float* a = (float*)_mm_malloc(sizeof(float)*numElements, 64);
    __declspec(aligned(64))float res = 0.0;

    #pragma vector aligned
    #pragma ivdep
    for (int q = 0; q < numElements; q++)
    {
        a[q] = 1.0;
    }

    #pragma omp parallel num_threads(4) shared(res)
    {
        int offset = worksize * omp_get_thread_num();

        __m512 b_0 = _mm512_setzero();//KNF, cheaply emulated to KNC
        __m512 b_1 = _mm512_setzero();
        __m512 b_2 = _mm512_setzero();
        __m512 b_3 = _mm512_setzero();
        __m512 b_4 = _mm512_setzero();
        __m512 b_5 = _mm512_setzero();
        __m512 b_6 = _mm512_setzero();
        __m512 b_7 = _mm512_setzero();

        for (int m = 0; m < reps; m++)
        {
            #pragma vector aligned
            #pragma ivdep
            for (int q = offset; q < offset+worksize; q+=128)
            {
                // KNF, emulated on KNC
                __m512 a_0 = _mm512_load_ps(&(a[q]));
                __m512 a_1 = _mm512_load_ps(&(a[q+16]));
                __m512 a_2 = _mm512_load_ps(&(a[q+32]));
                __m512 a_3 = _mm512_load_ps(&(a[q+48]));
                __m512 a_4 = _mm512_load_ps(&(a[q+64]));
                __m512 a_5 = _mm512_load_ps(&(a[q+80]));
                __m512 a_6 = _mm512_load_ps(&(a[q+96]));
                __m512 a_7 = _mm512_load_ps(&(a[q+112]));

                // Works for both KNF and KNC
                b_0 = _mm512_add_ps(b_0, a_0);
                b_1 = _mm512_add_ps(b_1, a_1);
                b_2 = _mm512_add_ps(b_2, a_2);
                b_3 = _mm512_add_ps(b_3, a_3);
                b_4 = _mm512_add_ps(b_4, a_4);
                b_5 = _mm512_add_ps(b_5, a_5);
                b_6 = _mm512_add_ps(b_6, a_6);
                b_7 = _mm512_add_ps(b_7, a_7);
            }
            // Works for both KNF and KNC
            b_0 = _mm512_add_ps(b_0, b_1);
            b_2 = _mm512_add_ps(b_2, b_3);
            b_4 = _mm512_add_ps(b_4, b_5);
            b_6 = _mm512_add_ps(b_6, b_7);
            b_0 = _mm512_add_ps(b_0, b_2);
            b_4 = _mm512_add_ps(b_4, b_6);
            b_0 = _mm512_add_ps(b_0, b_4);
        }

        #pragma omp critical
        {
            res += _mm512_reduce_add_ps(b_0);// Common for both KNF and KNC
        }
    }
    _mm_free(a);
    return res;
#else
    return 0.0;
#endif
}

float __declspec(target(mic)) testIntrinsics_write_caches(const int reps, 
        const int eversion, const float value, const int worksize)
{
#ifdef __MIC__ || __MIC2__

    size_t numElements;
    numElements = worksize*4;
    float* a = (float*)_mm_malloc(sizeof(float)*numElements, 64);
    __declspec(aligned(64))float res = 0.0;

    #pragma omp parallel num_threads(4) shared(res)
    {
        int offset = worksize * omp_get_thread_num();
        __declspec(aligned(64))float writeData = value + 
            static_cast<float>(omp_get_thread_num());

        __m512 toWrite = _mm512_load_ps(&(writeData));
        __m512 one = _mm512_set_1to16_ps(1.0);

        for (int m = 0; m < reps; m++)
        {
            #pragma vector aligned
            #pragma ivdep
            for (int q = offset; q < offset+worksize; q+=128)
            {
                __m512 a_0 = toWrite;
                __m512 a_1 = toWrite;
                __m512 a_2 = toWrite;
                __m512 a_3 = toWrite;
                __m512 a_4 = toWrite;
                __m512 a_5 = toWrite;
                __m512 a_6 = toWrite;
                __m512 a_7 = toWrite;

                // For KNF, cheaply converted on KNC
                _mm512_store_ps(&(a[q]), a_0);
                _mm512_store_ps(&(a[q+16]), a_1);
                _mm512_store_ps(&(a[q+32]), a_2);
                _mm512_store_ps(&(a[q+48]), a_3);
                _mm512_store_ps(&(a[q+64]), a_4);
                _mm512_store_ps(&(a[q+80]), a_5);
                _mm512_store_ps(&(a[q+96]), a_6);
                _mm512_store_ps(&(a[q+112]), a_7);
            }
            toWrite = _mm512_add_ps(toWrite, one); // Common for KNF and KNC
        }
    }
    // Sum something in a to avoid compiler optimizations
    res = a[0] + a[numElements-1];
    _mm_free(a);
    return res;
#else
    return 0.0;
#endif
}
