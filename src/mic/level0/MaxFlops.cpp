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
#include <omp.h>
#include "MaxFlops.h"
#include "OptionParser.h"
#include "ProgressBar.h"
#include "ResultDatabase.h"
#include "Timer.h"

// Forward declarations
template <class T>
void RunTest(ResultDatabase &resultDB, int npasses, int verbose, int quiet,
    float repeatF, ProgressBar &pb, const char* precision, const int micdev);

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
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    // No specific options for this benchmark
}

// ****************************************************************************
// Function: checkResults
//
// Purpose:
//   Checks the results of the floating point test for consistency by comparing
//   the first half of memory to the second half.
//
// Arguments:
//   hostMem - host memory containing the result of the FP test
//   numFloats - length of hostMem array
//
// Returns:  nothing
// 
// Modifications:
//
// ****************************************************************************
template <class T>
void CheckResults(const T* hostMem, const int numFloats)
{
    const int halfNumFloats = numFloats / 2;
    for (int j=0 ; j<halfNumFloats; ++j)
    {
        if (hostMem[j] != hostMem[numFloats-j-1])
        {
            cout << "Error; hostMem[" << j << "]=" << hostMem[j]
                << " is different from its twin element hostMem["
                << (numFloats-j-1) << "]=" << hostMem[numFloats-j-1]
                <<"; stopping check\n";
            break;
        }
    }
}

// ****************************************************************************
// Function: initData
//
// Purpose:
//   Randomly intialize the host data in preparation for the FP test.
//
// Arguments:
//   hostMem - uninitialized host memory
//   numFloats - length of hostMem array
//
// Returns:  nothing
//
// Programmer: Zhi Ying(zhi.ying@intel.com)
//             Jun Jin(jun.i.jin@intel.com)
//
// Modifications:
//
// ****************************************************************************
template <class T>
void InitData(T *hostMem, const int numFloats)
{
    const int halfNumFloats = numFloats/2;
    srand((unsigned)time(NULL));
    for (int j=0; j<halfNumFloats; ++j)
    {
        hostMem[j] = hostMem[numFloats-j-1] = (T)((rand()/ (float)RAND_MAX) *
                10.0);
    }
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Measures the floating point capability of the device for a variety of 
//   combinations of arithmetic operations.
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Zhi Ying(zhi.ying@intel.com)
//             Jun Jin(jun.i.jin@intel.com)
//
// Creation: May 23, 2011
//
// Modifications:
// 12/12/12 - Kyle Spafford - Code style and minor integration updates
//
// ****************************************************************************
void RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
    const bool verbose = op.getOptionBool("verbose");
    // Quiet == no progress bar.
    const bool quiet   = op.getOptionBool("quiet");
    const unsigned int passes = op.getOptionInt("passes");
    const int micdev = op.getOptionInt("target");

    double repeatF = 3;
    cout << "Adjust repeat factor = " << repeatF << "\n";

    // Initialize progress bar
    int totalRuns = 16*passes*2;
    ProgressBar pb(totalRuns);
    if (!verbose && !quiet) 
    {
        pb.Show(stdout);
    }

    RunTest<float>(resultDB, passes, verbose, quiet,
                   repeatF, pb, "-SP", micdev);
    RunTest<double>(resultDB, passes, verbose, quiet,
                    repeatF, pb, "-DP", micdev);

    if (!verbose) cout << endl;
}

template <class T>
void RunTest(ResultDatabase &resultDB, const int npasses, const int verbose,
        const int noPB, const float repeatF, ProgressBar &pb, 
        const char* precision, const int micdev)
{
    char sizeStr[128];
    static __declspec(target(mic)) T *hostMem;

    int realRepeats = (int)round(repeatF*20);
    if (realRepeats < 2) realRepeats = 2;

    // Allocate host memory
    int halfNumFloats = 1024*1024;
    int numFloats = 2*halfNumFloats;
    hostMem = (T*)_mm_malloc(sizeof(T)*numFloats,64);

    sprintf (sizeStr, "Size:%07d", numFloats);
    float t = 0.0f;
    double TH;
    double flopCount;
    double gflop;

    for (int pass=0 ; pass<npasses ; ++pass)
    {
        ////////// Add1 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}

        int add1TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            Add1_MIC<T>(numFloats,hostMem, realRepeats, 10.0);
        }
        t = Timer::Stop(add1TimerHandle, "add1");
        flopCount = (double)numFloats * realRepeats * omp_get_num_threads();
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("Add1")+precision, sizeStr, "GFLOPS", gflop);
        
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// Add2 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int add2TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            Add2_MIC<T>(numFloats,hostMem, realRepeats, 10.0);
        }
        t = Timer::Stop(add2TimerHandle, "add2");
        flopCount = (double)numFloats * realRepeats * 120 * 2;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("Add2")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// Add4 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int add4TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            Add4_MIC<T>(numFloats,hostMem, realRepeats, 10.0);
        }
        t = Timer::Stop(add4TimerHandle, "add4");
        flopCount = (double)numFloats * realRepeats * 60 * 4;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("Add4")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// Add8 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int add8TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            Add8_MIC<T>(numFloats,hostMem, realRepeats, 10.0);
        }
        t = Timer::Stop(add8TimerHandle, "add8");
        flopCount = (double)numFloats * realRepeats * 80 * 3;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("Add8")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// Mul1 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int mul1TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            Mul1_MIC<T>(numFloats,hostMem, realRepeats, 1.01);
        }
        t = Timer::Stop(mul1TimerHandle, "mul1");
        flopCount = (double)numFloats * 2 * realRepeats * 200;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("Mul1")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// Mul2 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int mul2TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            Mul2_MIC<T>(numFloats,hostMem, realRepeats, 1.01);
        }
        t = Timer::Stop(mul2TimerHandle, "mul2");
        flopCount = (double)numFloats * 2 * realRepeats * 100 * 2;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("Mul2")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// Mul4 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int mul4TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            Mul4_MIC<T>(numFloats,hostMem, realRepeats, 1.01);
        }
        t = Timer::Stop(mul4TimerHandle, "mul4");
        flopCount = (double)numFloats * 2 * realRepeats * 50 * 4;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("Mul4")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// Mul8 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int mul8TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            Mul8_MIC<T>(numFloats,hostMem, realRepeats, 1.01);
        }
        t = Timer::Stop(mul8TimerHandle, "mul8");
        flopCount = (double)numFloats * 2 * realRepeats * 25 * 8;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("Mul8")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// MAdd1 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int madd1TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            MAdd1_MIC<T>(numFloats,hostMem, realRepeats, 10.0, 0.9899);
        }
        t = Timer::Stop(madd1TimerHandle, "madd1");
        flopCount = (double)numFloats * 2 * realRepeats * omp_get_num_threads() * 1;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("MAdd1")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// MAdd2 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int madd2TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            MAdd2_MIC<T>(numFloats,hostMem, realRepeats, 10.0, 0.9899);
        }
        t = Timer::Stop(madd2TimerHandle, "madd2");
        flopCount = (double)numFloats * 2 * realRepeats * 120 * 2;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("MAdd2")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// MAdd4 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int madd4TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            MAdd4_MIC<T>(numFloats,hostMem, realRepeats, 10.0, 0.9899);
        }
        t = Timer::Stop(madd4TimerHandle, "madd4");
        flopCount = (double)numFloats * 2 * realRepeats * 60 * 4;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("MAdd4")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// MAdd8 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int madd8TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            MAdd8_MIC<T>(numFloats,hostMem, realRepeats, 10.0, 0.9899);
        }
        t = Timer::Stop(madd8TimerHandle, "madd8");
        flopCount = (double)numFloats * 2 * realRepeats * 30 * 8;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("MAdd8")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// MulMAdd1 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int mulmadd1TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            MulMAdd1_MIC<T>(numFloats,hostMem, realRepeats, 3.75, 0.355);
        }
        t = Timer::Stop(mulmadd1TimerHandle, "mulmadd1");
        flopCount = (double)numFloats * 3 * realRepeats * 160 * 1;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("MulMAdd1")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// MulMAdd2 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int mulmadd2TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            MulMAdd2_MIC<T>(numFloats,hostMem, realRepeats, 3.75, 0.355);
        }
        t = Timer::Stop(mulmadd2TimerHandle, "mulmadd2");
        flopCount = (double)numFloats * 3 * realRepeats * 80 * 2;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("MulMAdd2")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// MulMAdd4 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int mulmadd4TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            MulMAdd4_MIC<T>(numFloats,hostMem, realRepeats, 3.75, 0.355);
        }
        t = Timer::Stop(mulmadd4TimerHandle, "mulmadd4");
        flopCount = (double)numFloats * 3 * realRepeats * 40 * 4;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("MulMAdd4")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);

        ////////// MulMAdd8 //////////
        InitData<T>(hostMem,numFloats);
        #pragma offload target(mic:micdev) in(hostMem:length(numFloats) free_if(0))
        {}
        int mulmadd8TimerHandle = Timer::Start();
        #pragma offload target(mic:micdev) in(numFloats,realRepeats) nocopy(hostMem)
        {
            MulMAdd8_MIC<T>(numFloats,hostMem, realRepeats, 3.75, 0.355);
        }
        t = Timer::Stop(mulmadd8TimerHandle, "mulmadd8");
        flopCount = (double)numFloats * 3 * realRepeats * 20 * 8;
        gflop = flopCount / (double)(t*1e9);
        resultDB.AddResult(string("MulMAdd8")+precision, sizeStr, "GFLOPS", gflop);
        #pragma offload target(mic:micdev) out(hostMem:length(numFloats) alloc_if(0))
        {}
        CheckResults<T>(hostMem,numFloats);
        pb.addItersDone();
        if (!verbose && !noPB)pb.Show(stdout);
    }
    _mm_free(hostMem);
}
