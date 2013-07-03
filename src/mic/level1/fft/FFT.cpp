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

#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mkl.h"

#include "fftlib.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"

using namespace std;

// Forward Declarations
template <class T2>
void RunTest(const string& name,
    ResultDatabase &resultDb, OptionParser &op);

void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("MB", OPT_INT, "0", "data size (in MiB)");
}

template <typename T2>
void init(T2 *source, int fftsz, int n_ffts)
{
#pragma omp parallel for
    for (int m = 0; m < n_ffts; ++m)
    for (int n = 0; n < fftsz;  ++n)
    {
        source[n + fftsz*m].x = cos((1.0+m)/fftsz * n);
        source[n + fftsz*m].y = sin((1.0+m)/fftsz * n);
    }
}

template <class T2>
__declspec(target(mic))
int checkDiff(T2 *source, int fftsz, int n_ffts)
{
    int diff = 0;
#pragma omp parallel for shared(source, diff)
    for (int m = 0; m < n_ffts; ++m)
    for (int n = 0; n < fftsz;  ++n)
    {
        T2 exd, got, scale, thr;
        scale.x = fftsz;
        got = source[n + fftsz*m];
        exd.x = scale.x * cos((1.0+m)/fftsz * n);
        exd.y = scale.x * sin((1.0+m)/fftsz * n);
        thr.x = 1e-6 * scale.x;
        if ( !(fabs(got.x - exd.x) < thr.x && fabs(got.y - exd.y) < thr.x) )
        {
            if (diff == 0)
            {
                printf("[%i,%i] expected (%lg,%lg) got (%lg,%lg)\n",
                       n,m,exd.x,exd.y,got.x,got.y);
            }
            diff = 1;
            break;
        }
    }
    return diff;
}

template <class T2>
void RunTest(const string& name, ResultDatabase &resultDB, OptionParser &op)
{
    static __declspec(target(mic)) T2 *source;
    int chk;
    unsigned long bytes = 0;
    const int micdev = op.getOptionInt("device");
    const bool verbose = op.getOptionBool("verbose");
    
    // Get problem size
    if (op.getOptionInt("MB") == 0) {
        int probSizes[4] = { 1, 8, 96, 256 };
        int sizeIndex = op.getOptionInt("size")-1;
        if (sizeIndex < 0 || sizeIndex >= 4) {
            cerr << "Invalid size index specified\n";
            exit(-1);
        }
        bytes = probSizes[sizeIndex];
    } else {
        bytes = op.getOptionInt("MB");
    }
    // Convert to MiB
    bytes *= 1024 * 1024;

    int passes = op.getOptionInt("passes");

    // The size of the transform computed is fixed at 512 complex elements
    int fftsz = 512;        
    int N = (bytes)/sizeof(T2);
    int n_ffts = N/fftsz;

    // Allocate space (aligned)
    source = (T2*) MKL_malloc(bytes,  4096);

    //allocate buffers and create FFT plans
    #pragma offload target(mic:micdev) in(fftsz, n_ffts)       \
                                       nocopy(source:length(N) \
                                       align(4096) alloc_if(1) free_if(0))
    {
        forward((T2*)NULL, fftsz, n_ffts);
        inverse((T2*)NULL, fftsz, n_ffts);
    }

    const char *sizeStr;
    stringstream ss;
    ss << "N=" << (long)N;
    sizeStr = strdup(ss.str().c_str());

    for(int k = 0; k < passes; k++)
    {
        init<T2>( source, fftsz, n_ffts );
        // Warmup
        if (k==0)
        {
            #pragma offload target(mic:micdev) in(fftsz, n_ffts)   \
                                               in(source:length(N) \
                                               alloc_if(0)  free_if(0))
            {
                forward(source, fftsz, n_ffts);
            }
        }

        // Time forward fft with data transfer over PCIe
        int fwd_pcie_timer_handle = Timer::Start();
        
        // Using in rather than inout to be consistent with CUDA version.
        #pragma offload target(mic:micdev) in(fftsz, n_ffts)               \
                                           in(source:length(N) alloc_if(0) \
                                           free_if(0))
        {
            forward(source, fftsz, n_ffts);
        }
        double time_fwd_pcie = Timer::Stop(fwd_pcie_timer_handle, "fwd_pcie");

        #pragma offload target(mic:micdev) out(source:length(N) alloc_if(0) \
                free_if(0))
        {
        }

        // Time inverse fft with data transfer over PCIe
        int rev_pcie_timer_handle = Timer::Start();
        #pragma offload target(mic:micdev) in(fftsz, n_ffts)   \
                                           in(source:length(N) \
                                           alloc_if(0)  free_if(0))
        {
            inverse(source, fftsz, n_ffts);
        }
        double time_inv_pcie = Timer::Stop(rev_pcie_timer_handle, "rev_pcie");

        #pragma offload target(mic:micdev) out(source:length(N) alloc_if(0) \
            free_if(0))
        {}

        // Check result
        #pragma offload target(mic:micdev) in(fftsz,n_ffts) nocopy(source) \
            out(chk)
        {
            chk = checkDiff(source, fftsz, n_ffts);
        }
        if (verbose || chk)
        {
            cout << "Test " << k << ((chk) ? ": Failed\n" : ": Passed\n");
        }

        // Time forward fft without data transfer
        int time_fwd_handle = Timer::Start();
        #pragma offload target(mic:micdev) in(fftsz, n_ffts) nocopy(source)
        {
            forward(source, fftsz, n_ffts);
        }
        double time_fwd_native = Timer::Stop(time_fwd_handle, "fwd");

        // Time inverse fft without data transfer
        int time_rev_handle = Timer::Start();
        #pragma offload target(mic:micdev) in(fftsz, n_ffts) nocopy(source)
        {
            inverse(source, fftsz, n_ffts);
        }
        double time_inv_native = Timer::Stop(time_rev_handle, "rev");

        // Calculate gflops
        double flop_count    = n_ffts*(5*fftsz*log2(fftsz));
        double GF_fwd_pcie   = flop_count / (time_fwd_pcie   * 1e9);
        double GF_fwd_native = flop_count / (time_fwd_native * 1e9);
        double GF_inv_pcie   = flop_count / (time_inv_pcie   * 1e9);
        double GF_inv_native = flop_count / (time_inv_native * 1e9);

        resultDB.AddResult(name, sizeStr, "GFLOPS", GF_fwd_native);
        resultDB.AddResult(name+"_PCIe", sizeStr, "GFLOPS", GF_fwd_pcie);
        resultDB.AddResult(name+"_Parity", sizeStr, "N", 
                (time_fwd_pcie - time_fwd_native) / time_fwd_native);

        resultDB.AddResult(name+"-INV", sizeStr, "GFLOPS", GF_inv_native);
        resultDB.AddResult(name+"-INV_PCIe", sizeStr, "GFLOPS", GF_inv_pcie);
        resultDB.AddResult(name+"-INV_Parity", sizeStr, "N", 
                (time_inv_pcie - time_inv_native) / time_inv_native);
    }

    // Cleanup FFT plans and buffers
    #pragma offload target(mic:micdev) nocopy(source:length(N) \
            alloc_if(0) free_if(1))
    {
        forward((T2*)NULL, 0, 0);
        inverse((T2*)NULL, 0, 0);
    }
    MKL_free(source);
}


void
RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
    const bool verbose = op.getOptionBool("verbose");

    if (verbose) // print MKL version info
    {
        static char mklver[200];
        char *p;
        MKL_Get_Version_String(mklver,sizeof(mklver));
        mklver[sizeof(mklver)-1] = 0;
        p = strrchr(mklver,' ');
        if (p) while (p[0]==' ' && p[1]==0) *p-- = 0;
        printf("SHOC FFT benchmark using MKL verison %s\n",mklver);
    }
    RunTest<cplxflt>("SP-FFT", resultDB, op);
    RunTest<cplxdbl>("DP-FFT", resultDB, op);
}

// Useful routine for debugging 
/*
template <class T2>
void maybe_dump(const char *legend, const T2 *source);
template <class T2>
void maybe_dump(const char *legend, const T2 *source)
{
    const int N = 30;
    const char *fmt;

    if (dp<T2>())
    {
        if (!dump_dp) return;
        fmt = "%2i: (%24.17lg %24.17lg)\n";
    }
    else
    {
        if (!dump_sp) return;
        fmt = "%2i: (%14.7g %14.7g)\n";
    }

    fprintf(stdout, "Dump of %s:\n", legend);
    for (int i = 0; i < N; i++)
    {
        fprintf(stdout, fmt, i, source[i].x, source[i].y);
    }
}
*/
