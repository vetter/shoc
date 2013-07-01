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
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include "offload.h"
#include "omp.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "sortKernel.h"
#include "Sort.h"
#include "Timer.h"

#ifdef TARGET_ARCH_LRB
#include <pthread.h>
#endif

#define ALIGN (4096)
#define BLOCK 768

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
    op.addOption("iterations", OPT_INT, "256", "specify scan iterations");
    op.addOption("nthreads", OPT_INT, "64", "specify number of threads");

}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the scan (parallel prefix sum) benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
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
RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
    int device;

    cout << "Running test with unsigned int" << endl;
    RunTest<unsigned int>("Sort", resultDB, op);

}

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{
    int probSizes[4] = { 1, 8, 48, 96 };

    int size = probSizes[op.getOptionInt("size")-1];
    
    // Convert to MiB
    size = (size*1024*1024)/sizeof(T);
    
    // Create input data on CPU
    unsigned int bytes = size * sizeof(T);

    // Allocate Host Memory
    __declspec(target(MIC)) static T *hkey, *outkey;
    __declspec(target(MIC)) static T *hvalue, *outvalue;

    hkey   = (T*)_mm_malloc(bytes,ALIGN);
    hvalue = (T*)_mm_malloc(bytes,ALIGN);

    outkey   = (T*)_mm_malloc(bytes,ALIGN);
    outvalue = (T*)_mm_malloc(bytes,ALIGN);


    // Initialize host memory
    cout << "Initializing host memory." << endl;

    srand(time(NULL));
    for (int i = 0; i < size; i++)
    {
        hkey[i] = hvalue[i]= (i+255) % 1089; // Fill with some pattern
    }

    int micdev = op.getOptionInt("target");
    int iters = op.getOptionInt("passes");
    int numThreads = op.getOptionInt("nthreads");

    cout << "nthreads   = " <<numThreads<< endl;

    cout << "Running benchmark" << endl;
    for(int it=0;it<iters;it++)
    {

        // Allocating buffer on card
        #pragma offload target(mic:micdev) in(hkey:length(size)  free_if(0)) \
                in(hvalue:length(size) free_if(0))\
                out(outkey:length(size) free_if(0))\
                out(outvalue:length(size) free_if(0))
        {
        }

        int txToDevTimerHandle = Timer::Start();
        // Get data transfer time
        #pragma offload target(mic:micdev) in(hkey:length(size) alloc_if(0) \
                free_if(0)) in(hvalue:length(size) alloc_if(0) free_if(0))
        {
        }
        double transferTime = Timer::Stop(txToDevTimerHandle, "tx to device");

        int kernelTimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) nocopy(hkey:length(size) \
                alloc_if(0) free_if(0))                             \
                nocopy(hvalue:length(size) alloc_if(0) free_if(0))  \
                nocopy(outkey:length(size) alloc_if(0) free_if(0))  \
                nocopy(outvalue:length(size) alloc_if(0) free_if(0))\
                in(numThreads)
        {
            sortKernel<T>(hkey, hvalue, outkey, outvalue, size, numThreads);
        }
        double totalRunTime = Timer::Stop(kernelTimerHandle, "sort");

        int txFromDevTimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) nocopy(hkey:length(size) \
                alloc_if(0) free_if(1))                             \
                nocopy(hvalue:length(size) alloc_if(0) free_if(1))  \
                out(outkey:length(size) alloc_if(0))                \
                out(outvalue:length(size) alloc_if(0))
        {
        }
        transferTime += Timer::Stop(txFromDevTimerHandle, "tx from device" );

        // If results aren't correct, don't report perf numbers
        if (!verifyResult<T>(outkey, outvalue, size))
        {
            return;
        }

        char atts[1024];
        double avgTime = (totalRunTime / (double) iters);
        sprintf(atts, "%d items", size);
        double gb = (double)(size * sizeof(T)) / (1000. * 1000. * 1000.);
        resultDB.AddResult(testName, atts, "GB/s", gb / avgTime);
        resultDB.AddResult(testName+"_PCIe", atts, "GB/s",
                gb / (avgTime + transferTime));
        resultDB.AddResult(testName+"_Parity", atts, "N",
                transferTime / avgTime);

    }
    // Clean up
    _mm_free(hkey);
    _mm_free(hvalue);

}

template <class T>
void radixoffset(T* hkey, T* tkey, const size_t n,const unsigned int iter)
{
    int i;
    #pragma omp parallel for private(i) shared(hkey,tkey)
    for(i=0;i<n;i++)
    {
        tkey[i]=!((hkey[i]&(1<<iter))>>iter);
    }
}

template <class T>
void rearrange(T** key, T** value, T** tkey, T** tvalue, T** tarr, 
               const size_t n)
{

    unsigned int i;
    unsigned int totalfalses=(*tkey)[n-1]+(*tvalue)[n-1];

#pragma omp parallel for private(i) shared(tkey,tvalue)
    for(i=0;i<n;i++)
    {
        unsigned int t;
        if((*tkey)[i]==0)
        {
            (*tkey)[i]=i-(*tvalue)[i]+totalfalses;
        }
        else
        {
            (*tkey)[i]=(*tvalue)[i];
        }
    }

#pragma omp parallel for private(i)             \
    shared(tarr,tkey,tvalue,key,value)
    for(i=0;i<n;i++)
    {
        (*tarr)[(*tkey)[i]]=(*key)[i];
        (*tvalue)[(*tkey)[i]]=(*value)[i];
    }

    T* temp;
    temp= *key;
    *key= *tarr;
    *tarr= temp;

    temp= *value;
    *value= *tvalue;
    *tvalue=temp;
}

template <class T>
__declspec(target(MIC)) void  scanArray(T *input,  T* output, const size_t n)
{
    int numblocks;
    numblocks=(int)ceil((double)n/BLOCK);
    float* ipblocksum=(float*)malloc(numblocks*sizeof(float));
    float* opblocksum=(float*)malloc(numblocks*sizeof(float));

    output[0]=0.0;

    #pragma omp parallel for shared(input,output)
    for(int i=0;i<numblocks;i++)
    {
        int offset=i*BLOCK;
        int end=(i==numblocks-1)?(n-offset):BLOCK;
        int j;

        output[offset]=0.0;
        for(j=1;j<end;j++)
        {
            output[offset+j]=output[offset+j-1]+input[offset+j-1];
        }
        ipblocksum[i]=output[offset+j-1]+input[offset+j-1];
    }

    opblocksum[0]=ipblocksum[0];
    for(int i=1;i<numblocks;i++)
    {
        opblocksum[i]=opblocksum[i-1]+ipblocksum[i];
    }

    #pragma omp parallel for shared(output,opblocksum)
    for(int i=1;i<numblocks;i++)
    {
        int offset=i*BLOCK;
        int end=(i==numblocks-1)?(n-offset):BLOCK;
        float value=opblocksum[i-1];
        for(int j=0;j<end;j++)
        {
            output[offset+j]=output[offset+j]+value;
        }
    }

    free(opblocksum);
    free(ipblocksum);
}

template <class T>
bool verifyResult(T *key,  T* val, const size_t size)
{
    bool passed = true;

    for (unsigned int i = 0; i < size-1; ++i)
    {
        if ((key[i]) > (key[i+1]))
        {
            passed = false;
        }
    }
    cout << "Test ";
    if (passed)
        cout << "Passed" << endl;
    else
        cout << "---FAILED---" << endl;
    return passed;
}
