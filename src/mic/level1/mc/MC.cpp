// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Kyle Spafford <kys@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
// Copyright (c) 2011, UT-Battelle, LLC
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
#include "Timer.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "MonteCarlo.h"
#include <iostream>
using namespace std;

#define SIMDALIGN 64


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


// ********************************************************
//  Function: RandFloat
//
//  Purpose:
//    Generate a random floating point number within a range low-high
//    strings using stringstream
//
//  Arguments:
//    low: the low end of the range
//    high: the high end of th range
//  Returns:  a random floating point value
//
//  Modifications:
//
//  ********************************************************

inline float RandFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

// ********************************************************
//  Function: CND
//
//  Purpose:
//    Polynomial approximation of cumulative normal distribution function
//
//
//  Arguments:
//    d: input variable whose CND has to be computed
//  Returns:  CDF value of the input
//
//  Modifications:
//
//  ********************************************************

double CND(double d){
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double cnd = RSQRT2PI * exp(- 0.5 * d * d) *
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

// ********************************************************
//  Function: BlackScholesFormula
//
//  Purpose:
//    Computes Provides Option Value based on Black Scholes Formula
//
//
//  Arguments:
//    sf : stock price
//    xf : option stike
//    Tf : Option years
//    Rf : Riskless rate
//    vf : Volatility rate
//
//  Returns:  CDF value of the input
//
//  Modifications:
//
//  ********************************************************
void BlackScholesFormula(
    double& callResult,
    double Sf, //Stock price
    double Xf, //Option strike
    double Tf, //Option years
    double Rf, //Riskless rate
    double Vf)  //Volatility rate
{
    double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

    double sqrtT = sqrt(T);
    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double    d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);

    double expRT = exp(- R * T);
    callResult   = (S * CNDD1 - X * expRT * CNDD2);
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
   op.addOption("validate", OPT_INT, "0", "Validate results");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the MonteCarlo benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
//
// Creation: March 13, 2010
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(OptionParser &op, ResultDatabase &resultDB)
{
    printf("Running single precision  version of MonteCarlo benchmark\n");
    RunTest<float, 16>("MC-SP_16", resultDB, op);

    printf("Running double precision verison of MonteCarlo benchmark\n");
    RunTest<double, 8>("MC-DP_8", resultDB, op);
}

__declspec(target(MIC) align(4096))   int OPT_N;

template <class real, int MAXVL>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{

    __declspec(target(MIC) align(4096)) static real
        *CallResultParallel,
        *CallConfidence,
        *StockPrice,
        *OptionStrike,
        *OptionYears;

    double sTime, eTime;
    int mem_size, rand_size, validate;

    validate = op.getOptionInt("validate");


    const int RAND_N = 1 << 18;

    int numCores = 60; //Assuming 60 core Xeon Phi card

    // Number of grid points (specified in header file)
    const int probSizes[4] = { 16, 32, 40, 64 };
    int sizeClass = op.getOptionInt("size") - 1;
    assert(sizeClass >= 0 && sizeClass < 4);

    sizeClass = probSizes[sizeClass]; //number of data inputs in thousands to the simulation,
    //64 indicates 64K data inputs
    //problem size - we run half as many options for double precision
    OPT_N = 2*512*sizeClass*numCores;// Problem size

    // Host variables
    // Malloc host memory
    mem_size = sizeof(real)*OPT_N;
    rand_size = sizeof(real)*RAND_N;

    CallResultParallel = (real *)_mm_malloc(mem_size, SIMDALIGN);
    CallConfidence     = (real *)_mm_malloc(mem_size, SIMDALIGN);
    StockPrice         = (real *)_mm_malloc(mem_size, SIMDALIGN);
    OptionStrike       = (real *)_mm_malloc(mem_size, SIMDALIGN);
    OptionYears        = (real *)_mm_malloc(mem_size, SIMDALIGN);

    // Initialize Test Problem
    for(int i = 0; i < OPT_N; i++)
    {
        CallResultParallel[i] = 0.0;
        CallConfidence[i]= -1.0;
        StockPrice[i]    = RandFloat(5.0f, 50.0f);
        OptionStrike[i]  = RandFloat(10.0f, 25.0f);
        OptionYears[i]   = RandFloat(1.0f, 5.0f);
    }

    unsigned int passes = op.getOptionInt("passes");
    double start;
    

    // Initialize  data on the coprocessor
    #pragma offload target (mic:0)                                     \
        in(StockPrice: length(OPT_N) alloc_if(1) free_if(0))           \
        in(OptionStrike:length(OPT_N) alloc_if(1) free_if(0))          \
        in(OptionYears : length(OPT_N)  alloc_if(1) free_if(0))        \
        out(CallResultParallel : length(OPT_N) alloc_if(1) free_if(0)) \
        out(CallConfidence : length(OPT_N) alloc_if(1) free_if(0))
    {
    }

    // Start the timer for the PCIe transfer
    int txToDevTimerHandle = Timer::Start();

    #pragma offload target (mic:0)                              \
        in(OPT_N)                                               \
        in(StockPrice: length(OPT_N) alloc_if(0) free_if(0))    \
        in(OptionStrike:length(OPT_N) alloc_if(0) free_if(0))   \
        in(OptionYears : length(OPT_N)  alloc_if(0) free_if(0))
    {
    }

    double transferTime = Timer::Stop(txToDevTimerHandle, "tx to dev");

    double kernelTime;
    double otransferTime;
    int kernelTimerHandle;
    int rxFromDevTimerHandle;

    // Now run the benchmark
    for (unsigned int i = 0; i < passes; i++)
    {

        printf("Pass = %d\r",i);

        kernelTimerHandle = Timer::Start();
        // Do the compute
        #pragma offload target (mic:0)                                       \
           nocopy(StockPrice: length(OPT_N) alloc_if(0) free_if(0))          \
           nocopy(OptionStrike:length(OPT_N) alloc_if(0) free_if(0))         \
           nocopy(OptionYears : length(OPT_N)  alloc_if(0) free_if(0))       \
           nocopy(CallResultParallel : length(OPT_N) alloc_if(0) free_if(0)) \
           nocopy(CallConfidence : length(OPT_N) alloc_if(0) free_if(0))
        {
            MonteCarlo(CallResultParallel,
                       CallConfidence,
                       StockPrice,
                       OptionStrike,
                       OptionYears, OPT_N);
        }    // offload compute section

        kernelTime=Timer::Stop(kernelTimerHandle, "kernel runtime");

        // Now copy the results back
        rxFromDevTimerHandle = Timer::Start();
        #pragma offload target(mic:0)\
          out(CallResultParallel : length(OPT_N) alloc_if(0) free_if(0))  \
          out(CallConfidence : length(OPT_N) alloc_if(0) free_if(0))
        {
        }
        
        otransferTime=Timer::Stop(rxFromDevTimerHandle, "rx from device");

        //Validate the result from previous Monte Carlo code
        if (validate)
        {
        double delta, sum_delta, sum_ref, L1norm, sumReserve;
        double CallMaster;

        sum_delta = 0;
        sum_ref   = 0;
        sumReserve = 0;

        for(int i = 0; i < OPT_N; i++)
        {
            BlackScholesFormula(CallMaster,
                (double) StockPrice[i],
                (double) OptionStrike[i],
                (double) OptionYears[i],
                (double) RISKFREE,
                (double) VOLATILITY);
            delta = fabs(CallMaster - CallResultParallel[i]);
            sum_delta += delta;
            sum_ref   += fabs(CallMaster);
            if(delta > 1e-6)
                sumReserve += CallConfidence[i] / delta;
        }
        sumReserve /= (double)OPT_N;
        L1norm = sum_delta / sum_ref;
        printf("L1 norm: %E\n", L1norm);
        printf("Average reserve: %f\n", sumReserve);

        printf("...freeing CPU memory.\n");
        printf((sumReserve > 1.0f) ? "PASSED\n" : "FAILED\n");
        }

        double optPerSec = (OPT_N/kernelTime);

        resultDB.AddResult(testName, toString(OPT_N) + "Options", "Options/Second",
                           OPT_N / kernelTime);
        resultDB.AddResult(testName + "_PCIe", toString(OPT_N) + "Options", "Options/Second",
                           OPT_N / (kernelTime + transferTime + otransferTime));
        resultDB.AddResult(testName + "_Parity", toString(OPT_N) + "Options", "N",
                           (transferTime + otransferTime) / kernelTime);
    }

    printf("\n");

    // Free the memory on the card
   #pragma offload target (mic:0)                                       \
        in(StockPrice: length(OPT_N) alloc_if(0) free_if(1))            \
        in(OptionStrike:length(OPT_N) alloc_if(0) free_if(1))           \
        in(OptionYears : length(OPT_N)  alloc_if(0) free_if(1))         \
        out(CallResultParallel : length(OPT_N) alloc_if(0) free_if(1))  \
        out(CallConfidence : length(OPT_N) alloc_if(0) free_if(1))
    {
    }

    //Free host memory;
    CallResultParallel = (real *)_mm_malloc(mem_size, SIMDALIGN);
    CallConfidence     = (real *)_mm_malloc(mem_size, SIMDALIGN);
    StockPrice         = (real *)_mm_malloc(mem_size, SIMDALIGN);
    OptionStrike       = (real *)_mm_malloc(mem_size, SIMDALIGN);
    OptionYears        = (real *)_mm_malloc(mem_size, SIMDALIGN);

    _mm_free(CallResultParallel);
    _mm_free(CallConfidence);
    _mm_free(StockPrice);
    _mm_free(OptionStrike);
    _mm_free(OptionYears);
}
