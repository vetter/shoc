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

#include <stdio.h>
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific command line argument parsing.
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    // No specific options for this benchmark.
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   Measures the bandwidth of the bus connecting the host processor to the
//   MIC.
//
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
// 12/12/12 - Kyle Spafford -- Updated to preliminary version for MIC 
//
// ****************************************************************************
__declspec(target(MIC)) float *hostMem=NULL;

#define ALIGN  (4096)

void RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
    const bool verbose = op.getOptionBool("verbose");

    // Sizes are in kb
    const int nSizes  = 17;
    int sizes[nSizes] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
        32768, 65536};
    
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;

    // Create host memory
    hostMem = (float*)_mm_malloc(numMaxFloats*sizeof(float),ALIGN);

    if(hostMem==NULL)
    {
        cerr << "Couldn't allocate CPU memory! \n";
        cerr << "Test failed." << endl;
        return;
    }
    // Initialize memory with some pattern.
    for (int i = 0; i < numMaxFloats; i++)
    {
        hostMem[i] = i % 77;
    }

    const unsigned int passes = op.getOptionInt("passes");
    int micdev = op.getOptionInt("device");

    // Allocate memory on the card
    #pragma offload target(mic:micdev) \
    nocopy(hostMem:length(numMaxFloats) alloc_if(1) free_if(0) align(ALIGN) ) 
    {
    }

    // Three passes, forward and backward both
    for (int pass = 0; pass < passes; pass++)
    {
        // Step through sizes forward on even passes and backward on odd
        for (int i = 0; i < nSizes; i++)
        {
            int sizeIndex;
            if ((pass % 2) == 0)
            {
                sizeIndex = i;
            }
            else
            {
                sizeIndex = (nSizes - 1) - i;
            }

            int nbytes = sizes[sizeIndex] * 1024;

            //  D->H test
            int txFromDevTimerHandle = Timer::Start();

            #pragma offload target(mic:micdev) \
            out(hostMem:length((1024*sizes[sizeIndex]/4)) \
            free_if(0) alloc_if(0)  )
            {
            }
            double t = Timer::Stop(txFromDevTimerHandle, "tx from dev");

            if (verbose)
            {
                cerr << "Size " << sizes[sizeIndex] << "k took " << t <<
                    " sec\n";
            }

            double speed = (double(sizes[sizeIndex]) * 1024 / 
                    (1000. * 1000. * 1000.)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "% 6dkB", sizes[sizeIndex]);
            resultDB.AddResult("ReadbackSpeed", sizeStr, "GB/sec", speed);
            resultDB.AddResult("ReadbackTime", sizeStr, "ms", t*1000);
        }
    }
    // Free memory allocated on the mic
    #pragma offload target(mic:micdev) \
    in(hostMem:length(numMaxFloats) alloc_if(0)  )
    {
    }

    // Cleanup
    _mm_free(hostMem);
}
