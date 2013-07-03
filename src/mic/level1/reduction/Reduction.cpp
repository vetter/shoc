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
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#include "omp.h"
#include "offload.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"

#ifdef __MIC2__
#include <pthread.h>
#endif

using namespace std;

// Forward Declaration
template <class T>
void RunTest(string, ResultDatabase &, OptionParser &);

// ****************************************************************************
// Function: reduceGold
//
// Purpose:
//   Simple cpu reduce routine to verify device results.  This could be
//   replaced with Kahan summation for better accuracy.
//
// Arguments:
//   data : the input data
//   size : size of the input data
//
// Returns:  sum of the data
//
// ****************************************************************************
template <class T>
T reduceGold(const T *data, int size)
{
    T sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += data[i];
    }
    return sum;
}


template <typename T>
__declspec(target(mic)) T reductionKernel(T *data, size_t size)
{
    T ret = 0.0;
    T intermed[512];

    int nThreads = omp_get_max_threads();
    int nPerThread = size / nThreads;

    #pragma omp parallel for
    for (int i = 0; i < nThreads; i++)
    {
        T loopres = 0.0;
        for (int j = i*nPerThread; j < i*nPerThread+nPerThread; j++)
            loopres += data[j];

        intermed[i] = loopres;
    }

    for (int i = 0; i < nThreads; i++)
    {
        ret += intermed[i];
    }

    // Rest of the array
    for(int i = nThreads * nPerThread; i < size; i++)
    {
        ret += data[i];
    }
    return ret;
}

template <typename T>
bool check(T result, T ref) {

    float diff = fabs(result - ref);

    float threshold = 1e-2;
    if (diff >= threshold * ref)            
    {
        cout << "Test: Failed\n";
        cout << "Diff: " << diff;
        exit(-1);
    }
    else
    {
        cout<< "Passed\n";
    }
    return true;
}

void addBenchmarkSpecOptions(OptionParser& op)
{
    op.addOption("iterations", OPT_INT, "256",
            "specify reduction iterations");
}

template <typename T>
void RunTest(string testName, ResultDatabase& resultDB, OptionParser& op) 
{
    __attribute__ ((target(mic))) T *indata  = NULL;
    __attribute__ ((target(mic))) T *outdata = NULL;

    const int micdev = op.getOptionInt("device");

    // Get Problem Size
    int probSizes[4] = { 4, 8, 32, 64 };
    int N = probSizes[op.getOptionInt("size")-1];
    N = (N * 1024 * 1024) / sizeof(T);

    indata = (T*) _mm_malloc(N * sizeof(T), (2*1024*1024));
    if (!indata) return;

    outdata = (T*)_mm_malloc(64 * sizeof(T), (2*1024*1024));
    if (!outdata) return;

    // Initialize Host Memory
    cout << "Initializing memory." << endl;
    for(int i = 0; i < N; i++)
    {
        indata[i] = i % 3; // Fill with some pattern
    }

    T ref = reduceGold(indata, N);
    const int passes     = op.getOptionInt("passes");
    const int iterations = op.getOptionInt("iterations");;

    // Test attributes
    char atts[1024];
    sprintf(atts, "%d_items",N);

    cout<< "Running Benchmark\n";

    for (int k = 0; k < passes; k++)
    {
        T result;

        #pragma offload target(mic:micdev) \
        in(outdata:length(64)  align(4*1024*1024) alloc_if(1) free_if(0))
        {
        }
        // Warm up
        #pragma offload target(mic:micdev) \
        in(indata:length(N)  align(4*1024*1024) alloc_if(1) free_if(0))
        {
        }

        int txToCardTimerHandle = Timer::Start();

        #pragma offload target(mic:micdev) \
        in(indata:length(N)  alloc_if(0) free_if(0))
        {
        }
        double transferTime = Timer::Stop(txToCardTimerHandle, "tx to device");

        int reductionTimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) nocopy(indata:length(N) \
                align(4*1024) alloc_if(0) free_if(0))
        {
            for (int j=0; j<iterations; j++) 
            {
                result = (T)reductionKernel(indata, N);
            }
        }

        double avgTime = Timer::Stop(reductionTimerHandle, "reduce") / (double)iterations;


        int txFromCardTimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) \
                out(outdata:length(64) alloc_if(0) free_if(1) )
        {
        }
        transferTime += Timer::Stop(txFromCardTimerHandle, "tx from device");

        check(result, ref);

        // Free buffer on card
        #pragma offload target(mic:micdev) nocopy(indata:length(N) \
                align(4*1024*1024) alloc_if(0) free_if(1))
        {
        }

        double gbytes = (double)(N*sizeof(T))/(1000.*1000.*1000.);
        resultDB.AddResult(testName, atts, "GB/s", gbytes / avgTime);
        resultDB.AddResult(testName+"_PCIe", atts, "GB/s", gbytes /
                (avgTime + transferTime));
        resultDB.AddResult(testName+"_Parity", atts, "N",
                transferTime / avgTime);
    }
    _mm_free( indata);
    _mm_free( outdata);
}

/*
 * Best performance with:
 * setenv MIC_ENV_PREFIX MIC
 * setenv MIC_OMP_NUM_THREADS 90
 * setenv MIC_KMP_AFFINITY balanced,granularity=fine
 */
void RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
    cout << "Running single precision test" << endl;
    RunTest<float>("Reduction", resultDB, op);

    cout << "Running double precision test" << endl;
    RunTest<double>("Reduction-DP", resultDB, op);
}

