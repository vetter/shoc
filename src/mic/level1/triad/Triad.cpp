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
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "offload.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"

void addBenchmarkSpecOptions(OptionParser &op)
{
    ;
}

__declspec(target(MIC))
void
Triad(const float* A,
        const float* B, 
        float* C,
        const float s,
        const int start,
        const int length,
        const int nThreads)
{
    int index = (int)((length/256) * nThreads);

    #pragma omp parallel for
    #pragma vector aligned
    #pragma vector nontemporal
    for (int idx=start; idx<start+index; idx++)
    {
        C[idx] = A[idx] + s*B[idx];
    }

    #pragma omp parallel for
    #pragma ivdep
    for (int idx=start+index; idx<start+length; idx++)
    {
        C[idx] = A[idx] + s*B[idx];
    }
}


#define ALIGNMENT 4096

__declspec(target(MIC) align(4096)) float *A0, *B0, *C0,*A1, *B1, *C1;

void RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
    const bool verbose = op.getOptionBool("verbose");
    const int n_passes = op.getOptionInt("passes");
    const int micdev = op.getOptionInt("device");

    const int nSizes = 9;
    const size_t blockSizes[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        16384 };
    const size_t memSize =  blockSizes[nSizes - 1];
    int  numMaxFloats = 1024 * memSize / sizeof(float);
    int  halfNumFloats = numMaxFloats / 2;
    int nThreads = omp_get_max_threads_target( TARGET_MIC, micdev );

    __declspec(target(MIC)) float *h_mem;
    h_mem = (float *) _mm_malloc(sizeof(float)*numMaxFloats,ALIGNMENT);

    A0 =  (float *)_mm_malloc( blockSizes[nSizes - 1] * 1024, ALIGNMENT);
    B0 =  (float *)_mm_malloc( blockSizes[nSizes - 1] * 1024, ALIGNMENT);
    C0 =  (float *)_mm_malloc( blockSizes[nSizes - 1] * 1024, ALIGNMENT);

    float *A0dummy, *B0dummy,  *C0dummy, *A1dummy, *B1dummy, * C1dummy;
    A0dummy = A0;
    B0dummy = B0;
    C0dummy = C0;
    A1 =  ( float *)_mm_malloc( blockSizes[nSizes - 1] * 1024, ALIGNMENT);
    B1 =  ( float *)_mm_malloc( blockSizes[nSizes - 1] * 1024, ALIGNMENT);
    C1 =  ( float *)_mm_malloc( blockSizes[nSizes - 1] * 1024, ALIGNMENT);
    A1dummy = A1;
    B1dummy = B1;
    C1dummy = C1;


    #pragma offload target(MIC:micdev) \
    in(A0:length(numMaxFloats) free_if (0) alloc_if (1) align(4096)) \
    in(B0:length(numMaxFloats) free_if (0) alloc_if (1) align(4096)) \
    inout(C0:length(numMaxFloats) free_if (0) alloc_if (1) align(4096))
    { }

    #pragma offload target(MIC:micdev) \
    in(A1:length(numMaxFloats) free_if (0) alloc_if (1) align(4096)) \
    in(B1:length(numMaxFloats) free_if (0) alloc_if (1) align(4096)) \
    inout(C1:length(numMaxFloats) free_if (0) alloc_if (1) align(4096))
    { }

    float scalar = 1.75f;
    char sizeStr[256];
    
    for (int pass = 0; pass < n_passes; ++pass)
    {
        for (int i = 0; i < nSizes ; ++i)
        {
            int elemsInBlock = blockSizes[i] * 1024 / sizeof(float);
            for (int j = 0; j < halfNumFloats; ++j)
            {
                h_mem[j] = h_mem[halfNumFloats + j]
                    = (float) (drand48() * 10.0);
            }

            memcpy(A0, (void const*) h_mem, sizeof(float)*numMaxFloats);
            memcpy(B0, (void const*) h_mem, sizeof(float)*numMaxFloats);
            memcpy(A1, (void const*) h_mem, sizeof(float)*numMaxFloats);
            memcpy(B1, (void const*) h_mem, sizeof(float)*numMaxFloats);

            if (verbose)
            {
                cout << ">> Executing Triad: "
                     << "vector length: " << numMaxFloats 
                     << ", block size: " << elemsInBlock << " elements" 
                     << ", nThreads: " << nThreads
                     << std::endl;
            }
            sprintf(sizeStr, "Block:%05ldKB", blockSizes[i]);

            int crtIdx = 0;

            int kernelTimerHandle = Timer::Start();

            #pragma offload target(MIC:micdev) \
            in(A0:length(elemsInBlock) free_if (0) alloc_if (0) ) \
            in(B0:length(elemsInBlock) free_if (0) alloc_if (0) ) \
            nocopy(C0:free_if (0) alloc_if (0))
            {
                fflush(0);
                Triad(A0, B0, C0,  scalar, crtIdx, elemsInBlock, nThreads);
            }

            #pragma offload_transfer target(mic:micdev) \
            out(C0 [0:elemsInBlock]:alloc_if (0) free_if (0)) signal(C0)

            if (elemsInBlock < numMaxFloats)
            {
                // start downloading data for next block
                #pragma offload_transfer target(MIC:micdev) \ 
                in(A1[elemsInBlock:elemsInBlock]: free_if (0) alloc_if (0)) \
                in(B1[elemsInBlock:elemsInBlock]: free_if (0) alloc_if (0)) \
                signal(A1)
            }
            int blockIdx = 1;
            unsigned int currStream = 1;

            while (crtIdx < numMaxFloats)
            {
                currStream = blockIdx & 1;
                if (currStream)
                {
                    #pragma offload_wait target(mic:micdev) wait(C0)
                }

                else
                {
                    #pragma offload_wait target(mic:micdev) wait(C1)
                }

                crtIdx += elemsInBlock;
                if (crtIdx < numMaxFloats)
                {
                    if (currStream)
                    {
                        #pragma offload target(MIC:micdev) \
                        nocopy(A1,B1) wait(A1) nocopy(C1)
                        {
                            Triad(A1, B1, C1,  scalar, crtIdx, elemsInBlock, nThreads);
                        }
                        #pragma offload_transfer target(mic:micdev)                 \
                        out(C1[crtIdx:elemsInBlock]:alloc_if (0) free_if (0) ) \
                        signal(C1)
                    }
                    else
                    {
                        #pragma offload target(MIC:micdev) \
                        nocopy(A0,B0) wait(A0) nocopy(C0)
                        {
                            Triad(A0, B0, C0,  scalar, crtIdx, elemsInBlock, nThreads);
                        }

                        #pragma offload_transfer target(mic:micdev)                \
                        out(C0[crtIdx:elemsInBlock]:alloc_if (0) free_if (0) )\
                        signal(C0)
                    }
                }
               
                if (crtIdx+elemsInBlock < numMaxFloats)
                {
                    if (currStream)
                    {
                        #pragma offload_transfer target(MIC:micdev)            \
                        in(A0[crtIdx+elemsInBlock:elemsInBlock]: free_if (0)   \
                             alloc_if (0))                                     \
                        in(B0[crtIdx+elemsInBlock:elemsInBlock]: free_if (0)   \
                             alloc_if (0)) signal(A0)
                    }
                    else
                    {
                        #pragma offload_transfer target(MIC:micdev)            \
                        in(A1[crtIdx+elemsInBlock:elemsInBlock]: free_if (0)   \
                            alloc_if (0))                                      \
                        in(B1[crtIdx+elemsInBlock:elemsInBlock]: free_if (0)   \
                            alloc_if (0)) signal(A1)
                    }
                }
                blockIdx += 1;
                currStream = !currStream;
            } // end while

            double time = Timer::Stop(kernelTimerHandle, "triad");
            double triadFlops = ((double)numMaxFloats * 2.0) / (time*1e9);
            resultDB.AddResult("TriadFlops", sizeStr, "GFLOP/s", triadFlops);

            double bdwth = ((double)numMaxFloats*sizeof(float)*3.0)
                / (time*1000.*1000.*1000.);
            resultDB.AddResult("TriadBdwth", sizeStr, "GB/s", bdwth);
            fflush(stdout);

            if (verbose) cout << ">> checking memory\n";
            for (int j=0; j<halfNumFloats; ++j)
            {
                if (h_mem[j] != h_mem[j+halfNumFloats])
                {
                    fflush(stdout);
                    cout << "Error; hostMem[" << j << "]=" << h_mem[j]
                        << " is different from its twin element hostMem["
                        << (j+halfNumFloats) << "]: "
                        << h_mem[j+halfNumFloats] << "stopping check\n";
                    break;
                }
            }

            if (verbose)  printf("finish!\n");
            for (int j=0; j<numMaxFloats; ++j)
                h_mem[j] = 0.0f;

        } // end for
    } // end for

    #pragma offload target(MIC:micdev)                   \
    in(A0:length(numMaxFloats) alloc_if (0) free_if (1)) \
    in(B0:length(numMaxFloats) alloc_if (0) free_if (1)) \
    out(C0:length(numMaxFloats) alloc_if (0) free_if (1))
    { }

    #pragma offload target(MIC:micdev)                   \
    in(A1:length(numMaxFloats) alloc_if (0) free_if (1)) \
    in(B1:length(numMaxFloats) alloc_if (0) free_if (1)) \
    out(C1:length(numMaxFloats) alloc_if (0) free_if (1))
    { }

    // Cleanup
    A0=A0dummy;
    B0=B0dummy;
    C0=C0dummy;
    A1=A1dummy;
    B1=B1dummy;
    C1=C1dummy;
    _mm_free(h_mem);
    _mm_free(A0);
    _mm_free(B0);
    _mm_free(C0);
    _mm_free(A1);
    _mm_free(B1);
    _mm_free(C1);
}

