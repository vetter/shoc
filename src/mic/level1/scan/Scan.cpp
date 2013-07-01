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

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "omp.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"
 
#ifdef __MIC2__
#include <immintrin.h>
#endif 
 
#include "Scan.h"

using namespace std;

#define BLOCK 768
#define ERR 1.0e-4
#define NUM_THREADS (240)
#define ALIGN (4096)
// Last tuned for KNC hardware.
#define KNC_IDEAL_L2_BUFFER (256*16) * 1024  
__declspec(target(mic)) int ideal_buffer_size = KNC_IDEAL_L2_BUFFER;
#include "Scan_Kernel.h"

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
    cout << "Running single precision test" << endl;
    RunTest<float>("Scan", resultDB, op);

    // Test to see if this device supports double precision
    cout << "Running double precision test" << endl;
    RunTest<double>("Scan-DP", resultDB, op);
}

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{
    // Additional problem sizes for scaling
    int probSizes[8] = { 1, 8, 32, 64 , 128 , 256 , 512 };

    int size = probSizes[op.getOptionInt("size")-1];
    
    // Convert to MB
    size = (size *1024*1024)/sizeof(T);
    
    unsigned int bytes = size * sizeof(T);
    int micdev = op.getOptionInt("target");

    // Allocate Host Memory
    __declspec(target(MIC)) static T* h_idata;
    __declspec(target(MIC)) static T* reference;
    __declspec(target(MIC)) static T* h_odata;

    h_idata   = (T*)_mm_malloc(bytes,ALIGN);
    reference = (T*)_mm_malloc(bytes,ALIGN);
    h_odata   = (T*)_mm_malloc(bytes,ALIGN);
       
    // Initialize host memory
    for (int i = 0; i < size; i++) 
    { 
        h_idata[i] = i % 3; // Fill with some pattern
        h_odata[i] = i % 3;
        reference[i]=0.0;
    }


    // Allocate data to mic
    #pragma offload target(mic:micdev) in(h_idata:length(size) free_if(0)) \
        out(h_odata:length(size) free_if(0))
    {
    }

    int txToCardTimerHandle = Timer::Start();
    // Get data transfer time
    #pragma offload target(mic:micdev) in(h_idata:length(size) alloc_if(0) \
            free_if(0)) out(h_odata:length(size) alloc_if(0) free_if(0))
    {
    }
    double transferTime = Timer::Stop(txToCardTimerHandle, "tx to device");

    int passes = op.getOptionInt("passes");
    int iters = op.getOptionInt("iterations");

    // cout << "Running benchmark with size " << size << endl;
    for (int k = 0; k < passes; k++)
    {
      
        int kernelTimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) nocopy(h_idata:length(size) \
                alloc_if(0) free_if(0)) nocopy(h_odata:length(size)    \
                alloc_if(0) free_if(0))
        {
            
        int ThreadCount = NUM_THREADS;
        T* ipblocksum = (T*)malloc((ThreadCount+1)*sizeof(T));
        for (int j = 0; j < iters; j++)
        {    
          scanTiling<T>(h_idata, h_odata, size, ipblocksum, ThreadCount);
        }
        free ((void *)ipblocksum);
        }

        double totalScanTime = Timer::Stop(kernelTimerHandle, "scan");
   
        int txFromCardTimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) out(h_odata:length(size) \
                alloc_if(0) free_if(0))
        {
        }
        transferTime += Timer::Stop(txFromCardTimerHandle, "tx from device");
    
        // If results aren't correct, don't report perf numbers    
        if (! scanCPU<T>(h_idata, reference, h_odata, size))
        {    
            return;        
        }

        char atts[1024];
        double avgTime = (totalScanTime / (double) iters);
        sprintf(atts, "%d items", size);
        double gb = (double)(size * sizeof(T)) / (1000. * 1000. * 1000.);
        resultDB.AddResult(testName, atts, "GB/s", gb / avgTime);
        resultDB.AddResult(testName+"_PCIe", atts, "GB/s",
            gb / (avgTime + transferTime));
        resultDB.AddResult(testName+"_Parity", atts, "N",
            transferTime / avgTime);
    }
    
    // Clean up
    #pragma offload target(mic:micdev) in(h_idata:length(size) alloc_if(0) ) \
                                out(h_odata:length(size) alloc_if(0))
    {
    }
    _mm_free(h_idata);
    _mm_free(h_odata);
    _mm_free(reference);
}

// ****************************************************************************
// Function: scanCPU
//
// Purpose:
//   Simple cpu scan routine to verify device results
//
// Arguments:
//   data : the input data
//   reference : space for the cpu solution
//   dev_result : result from the device
//   size : number of elements
//
// Returns:  nothing, prints relevant info to stdout
//
// Modifications:
//
// ****************************************************************************
template <class T>
bool scanCPU(T *data, T* reference, T* dev_result, const size_t size)
{
    reference[0] = 0;
    bool passed = true;
    
    // NB: You cannot validate beyond a certain buffer size because
    // of rounding errors.
    if (size > 128) 
    {
        // This is an inclusive scan while the OpenMP code is an exclusive scan
        reference[0] = data[0];
        for (unsigned int i = 1; i < size; ++i)
        {
            reference[i] = data[i] + reference[i - 1];
        }
        
        for (unsigned int i = 0; i < size; ++i)
        {
            if (abs(reference[i] - dev_result[i]) > ERR )
            {
#ifdef VERBOSE_OUTPUT
                cout << "Mismatch at i: " << i << " ref: " << reference[i - 1]
                     << " dev: " << dev_result[i] << endl;
#endif
                passed = false;
            }
        }
    }
    cout << "Test ";
    if (passed)
        cout << "Passed" << endl;
    else
        cout << "Failed" << endl;
    return passed;
}
