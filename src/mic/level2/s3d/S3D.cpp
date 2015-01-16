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
#include <string>
#include <sstream>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "S3D.h"
#include "Timer.h"

#include "qssa_i.h"
#include "rdsmh_i.h"
#include "ratt_i.h"
#include "ratx_i.h"
#include "rdwdot_i.h"
#include "getrates_i_c.h"

using namespace std;

// Forward declaration
template <class real, int MAXVL>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

// ********************************************************
// Function: toString
//
// Purpose:
//   Simple templated function to convert objects into
//   strings using stringstream
//
// Arguments:
//   t: the object to convert to a string
//
// Returns:  a string representation
//
// Modifications:
//
// ********************************************************
template<class T> inline string toString(const T& t)
{
    stringstream ss;
    ss << t;
    return ss.str();
}

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
// Creation: March 13, 2010
//
// Modifications:
//
// ****************************************************************************
    void
addBenchmarkSpecOptions(OptionParser &op)
{
    ; // No S3D specific options
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the S3D benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: March 13, 2010
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
    RunTest<float, 16>("S3D-SP", resultDB, op); 
    RunTest<double, 8>("S3D-DP", resultDB, op);
}

#define gridarr_G(name,i,j) (name)[i-1+(n)*(j-1)]

#undef P
#undef T
#undef Y
#undef WDOT
#undef rr_r1
#undef ptemp
#undef ttemp
#undef yspec
#define P(i)       oneDarr(host_p,i)
#define T(i)       oneDarr(host_t,i)
#define Y(i,j)     gridarr_G(host_y,i,j)
#define WDOT(i,j)  gridarr_G(host_wdot,i,j)
#define rr_r1(i,j) vecarr(rr_r1,i,j)
#define ptemp(i)   oneDarr(ptemp,i)
#define ttemp(i)   oneDarr(ttemp,i)
#define yspec(i,j) vecarr(yspec,i,j)

template <class real, int MAXVL>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{
    // Number of grid points (specified in header file)
    /*const int probSizes[4] = { 16, 32, 40, 64 };
    int sizeClass = op.getOptionInt("size") - 1;
    assert(sizeClass >= 0 && sizeClass < 4);
    sizeClass = probSizes[sizeClass];
    int n = sizeClass * sizeClass * sizeClass;*/
    
    int probSizes_SP[4] = { 24, 32, 40, 48};
    int probSizes_DP[4] = { 16, 24, 32, 40};
    int *probSizes = (sizeof(real) == sizeof(double)) ? probSizes_DP : probSizes_SP;
    int sizeClass = op.getOptionInt("size") - 1;
    assert(sizeClass >= 0 && sizeClass < 4);
    int size = probSizes[sizeClass];
    int n = size * size * size;


    // Host variables
    __declspec(target(MIC) align(4096)) static real* host_t;
    __declspec(target(MIC) align(4096)) static real* host_p;
    __declspec(target(MIC) align(4096)) static real* host_y;
    __declspec(target(MIC) align(4096)) static real* host_wdot;
    __declspec(target(MIC) align(4096)) static real* host_molwt;

    __declspec(target(MIC) align(4096)) static real *host_rf;
    __declspec(target(MIC) align(4096)) static real *host_rb;
    __declspec(target(MIC) align(4096)) static real *host_rklow;
    __declspec(target(MIC) align(4096)) static real *host_c;
    __declspec(target(MIC) align(4096)) static real *host_a;
    __declspec(target(MIC) align(4096)) static real *host_eg;

    // Malloc host memory
    host_t=(real*)_mm_malloc(n*sizeof(real),ALIGN);
    host_p=(real*)_mm_malloc(n*sizeof(real),ALIGN);;
    host_y=(real*)_mm_malloc(Y_SIZE*n*sizeof(real),ALIGN);;
    host_wdot=(real*)_mm_malloc(WDOT_SIZE*n*sizeof(real),ALIGN);
    host_molwt=(real*)_mm_malloc(WDOT_SIZE*n*sizeof(real),ALIGN);

    host_rf=(real*)_mm_malloc(n*RF_SIZE*sizeof(real), ALIGN);
    host_rb=(real*)_mm_malloc(n*RB_SIZE*sizeof(real), ALIGN);
    host_rklow=(real*)_mm_malloc(n*RKLOW_SIZE*sizeof(real), ALIGN);
    host_c=(real*)_mm_malloc(n*C_SIZE*sizeof(real), ALIGN);
    host_a=(real*)_mm_malloc(n*A_SIZE*sizeof(real), ALIGN);
    host_eg=(real*)_mm_malloc(n*EG_SIZE*sizeof(real), ALIGN);

    // Initialize Test Problem

    // For now these are just 1, to compare results between cpu & host
    real rateconv = 1.0;
    real tconv = 1.0;
    real pconv = 1.0;

    // Initialize temp and pressure
    for (int i=0; i<n; i++)
    {
        host_p[i] = 1.0132e6;
        host_t[i] = 1000.0;
    }

    // Init molwt: for now these are just 1, to compare results betw. cpu & host
    for (int i=0; i<WDOT_SIZE; i++)
    {
        host_molwt[i] = 1;
    }

    // Initialize mass fractions
    for (int j=0; j<Y_SIZE; j++)
    {
        for (int i=0; i<n; i++)
        {
            host_y[(j*n)+i]= 0.0;
            if (j==14)
                host_y[(j*n)+i] = 0.064;
            if (j==3)
                host_y[(j*n)+i] = 0.218;
            if (j==21)
                host_y[(j*n)+i] = 0.718;
        }
    }

    unsigned int passes = op.getOptionInt("passes");
    double start;

    // Allocate data on the coprocessor
    #pragma offload target(mic:0)                                   \
        in(host_t:length(n) alloc_if(1) free_if(0))                 \
        in(host_p:length(n) alloc_if(1) free_if(0))                 \
        in(host_y:length(n*Y_SIZE)  alloc_if(1) free_if(0))         \
        in(host_molwt:length(n*WDOT_SIZE)  alloc_if(1) free_if(0))  \
        out(host_wdot:length(n*WDOT_SIZE)  alloc_if(1) free_if(0))  \
        in(host_rf:length(n*RF_SIZE)  alloc_if(1) free_if(0))       \
        in(host_rb:length(n*RB_SIZE)  alloc_if(1) free_if(0))       \
        in(host_rklow:length(n*RKLOW_SIZE)  alloc_if(1) free_if(0)) \
        in(host_c:length(n*C_SIZE)  alloc_if(1) free_if(0))         \
        in(host_a:length(n*A_SIZE)  alloc_if(1) free_if(0))         \
        in(host_eg:length(n*EG_SIZE)  alloc_if(1) free_if(0))
    {
    }   

    // Transfer the data
    fflush(0);
    int txToDevTimerHandle = Timer::Start();
    #pragma offload_transfer target(mic:0)                 \
        in(host_t:length(n) alloc_if(0) free_if(0))        \
        in(host_p:length(n) alloc_if(0) free_if(0))        \
        in(host_y:length(n*Y_SIZE) alloc_if(0) free_if(0)) \
        in(host_molwt:length(n*WDOT_SIZE) alloc_if(0) free_if(0))
    {
    }
    double transferTime = Timer::Stop( txToDevTimerHandle, "tx to dev");

    
    #pragma offload target(mic:0)                       \
        nocopy(host_p:length(n) alloc_if(0) free_if(0)) \
        nocopy(host_t:length(n) alloc_if(0) free_if(0))
    {
        ALIGN64 real dummy[n];
        #pragma omp parallel for private(dummy)
        for(int i = 0 ; i < n ; i++) 
        {
            host_p[i] = 1.0132e6;
            host_t[i] = 1000.0;
            dummy[i] = host_p[i]*host_t[i] + i;
        }   
    }

    // Now run the benchmark
    for (unsigned int i = 0; i < passes; i++)
    {

        int kernelTimerHandle = Timer::Start();

    // Do the compute 
        #pragma offload target(mic:0)                                  \
        nocopy(host_t:length(n) alloc_if(0) free_if(0))                \
        nocopy(host_p:length(n) alloc_if(0) free_if(0))                \
        nocopy(host_y:length(n*Y_SIZE) alloc_if(0) free_if(0))         \
        nocopy(host_molwt:length(n*WDOT_SIZE) alloc_if(0) free_if(0))  \
        nocopy(host_wdot:length(n*WDOT_SIZE) alloc_if(0) free_if(0))   \
        nocopy(host_rf:length(n*RF_SIZE) alloc_if(0) free_if(0))       \
        nocopy(host_rb:length(n*RB_SIZE) alloc_if(0) free_if(0))       \
        nocopy(host_rklow:length(n*RKLOW_SIZE) alloc_if(0) free_if(0)) \
        nocopy(host_c:length(n*C_SIZE) alloc_if(0) free_if(0))         \
        nocopy(host_a:length(n*A_SIZE) alloc_if(0) free_if(0))         \
        nocopy(host_eg:length(n*EG_SIZE) alloc_if(0) free_if(0))
        {
            ALIGN64 real rr_r1[MAXVL*22], yspec[MAXVL*22];
            ALIGN64 real ptemp[MAXVL], ttemp[MAXVL];
            ALIGN64 real  RCKWRK[1];
            ALIGN64 int ICKWRK[1];
            int m;

#pragma omp parallel for private(yspec, ptemp, ttemp, rr_r1)
            for (m = 1; m < n; m+=MAXVL) 
            {
                int i, j, ml, mu, nu;
                real rateconv, pconv, tconv,  molwt;
                mu = n;
                rateconv = 1.0;
                tconv = 1.0;
                pconv = 1.0;
                molwt = 1.0;
                //nu = min(MAXVL, mu-m+1);
		nu=(MAXVL<mu-m+1)?(MAXVL):(mu-m+1);
                for (i=1; i<=nu; i++)   ptemp(i) = P(m+i-1)*pconv;
                for (i=1; i<=nu; i++)   ttemp(i) = T(m+i-1)*tconv;

                for (i=1; i<=22; i++) 
                    for (j=1; j<=nu; j++) 
                        yspec(j, i) = Y(m+j-1,i);

                if (nu==MAXVL) 
                { 
                    getrates_i_VEC<real,MAXVL>(ptemp,ttemp,yspec,ICKWRK,RCKWRK,
                                               rr_r1);
                }
                else 
                {
                    getrates_i_<real,MAXVL>(ptemp,ttemp,yspec,&nu,ICKWRK,RCKWRK,
                                            rr_r1);
                }

                for (i=1; i<=22; i++) for (j=1; j<=nu; j++)
                    WDOT(m+j-1,i) = rr_r1(j,i)*rateconv*molwt;
            }
        }  // offload compute section

        double kernelTime = Timer::Stop(kernelTimerHandle, "s3d");

        // Now copy the results back
        int txFromDevTimerHandle = Timer::Start();
        #pragma offload target(mic:0)\
        out(host_wdot:length(n*WDOT_SIZE) alloc_if(0) free_if(0))
        {
        }
        double otransferTime = Timer::Stop(txFromDevTimerHandle, "tx from dev");

        double gflops = ((n*10000.) / 1.e9);

        resultDB.AddResult(testName, toString(n) + "_gridPoints", "GFLOPS",
                           gflops / kernelTime);
        resultDB.AddResult(testName + "_PCIe", toString(n) + "_gridPoints", "GFLOPS",
                           gflops / (kernelTime + transferTime + otransferTime));
        resultDB.AddResult(testName + "_Parity", toString(n) + "_gridPoints", "N",
                           (transferTime + otransferTime) / kernelTime);
    }


    //    // Print out answers for all wdot for loop_index of 0
    for (int i=0; i<WDOT_SIZE; i++) {
        printf("% 23.16E ", host_wdot[i*n]);
        if (i % 3 == 2)
            printf("\n");
    }
    printf("\n");

    // Free the memory on the card
#pragma offload target(mic:0)                           \
    in(host_t:length(n) alloc_if(0))                    \
    in(host_p:length(n) alloc_if(0))                    \
    in(host_y:length(n*Y_SIZE) alloc_if(0))             \
    in(host_molwt:length(n*WDOT_SIZE) alloc_if(0))      \
    out(host_wdot:length(n*WDOT_SIZE) alloc_if(0))      \
    in(host_rf:length(n*RF_SIZE) alloc_if(0))           \
    in(host_rb:length(n*RB_SIZE) alloc_if(0))           \
    in(host_rklow:length(n*RKLOW_SIZE) alloc_if(0))     \
    in(host_c:length(n*C_SIZE) alloc_if(0))             \
    in(host_a:length(n*A_SIZE) alloc_if(0))             \
    in(host_eg:length(n*EG_SIZE) alloc_if(0))
    {
    }


    //Free memory;
    _mm_free(host_t);
    _mm_free(host_p);
    _mm_free(host_y);
    _mm_free(host_wdot);
    _mm_free(host_molwt);

    _mm_free(host_rf);
    _mm_free(host_rb);
    _mm_free(host_rklow);
    _mm_free(host_c);
    _mm_free(host_a);
    _mm_free(host_eg);
}
