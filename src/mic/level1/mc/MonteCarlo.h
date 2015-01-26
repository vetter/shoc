// This example from an alpha release of the Scalable HeterOgeneous Computing
// (SHOC) Benchmark Suite Alpha v1.1.4a-mic for Intel MIC architecture
// Contact: Kyle Spafford <kys@ornl.gov>
//          Rezaur Rahman <rezaur.rahman@intel.com>
//
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

#pragma offload_attribute(push,target(mic))

#include "math.h"
#include "mkl_vsl.h"
#include "omp.h"
#define RANDSEED 123

const float RISKFREE = 0.06;
const float VOLATILITY = 0.10;

int getRngGaussian(const int method,
                   VSLStreamStatePtr stream,
                   const int n,
                   float *r,
                   float a,
                   float sigma)
{
     return vsRngGaussian (method, stream, n, r, a, sigma);
}

int getRngGaussian(const int method,
                   VSLStreamStatePtr stream,
                   const int n,
                   double *r,
                   double a,
                   double sigma)
{
    return vdRngGaussian (method, stream, n, r, a, sigma);
}


template <class real>
__attribute__((target(mic))) void MonteCarlo(real *h_CallResult,
                                             real *h_CallConfidence,
                                             real *S,
                                             real *X,
                                             real *T,
                                             int   OPT_N)
{

    __declspec(target(MIC) align(4096)) const int RAND_N = 1 << 18;

    __declspec(target(MIC) align(4096)) static const real  RVVLOG2E = (RISKFREE-0.5f*VOLATILITY*VOLATILITY)*M_LOG2E;
    __declspec(target(MIC) align(4096)) static const real  INV_RAND_N = 1.0f/RAND_N;
    __declspec(target(MIC) align(4096)) static const real  F_RAND_N = static_cast<real>(RAND_N);
    __declspec(target(MIC) align(4096)) static const real STDDEV_DENOM = 1 / (F_RAND_N * (F_RAND_N - 1.0f));
    __declspec(target(MIC) align(4096)) static const real CONFIDENCE_DENOM = 1 / sqrtf(F_RAND_N);
    __declspec(target(MIC) align(4096)) static const int BLOCKSIZE = 16*1024;
    __declspec(target(MIC) align(4096)) static const real  RLOG2E = RISKFREE*M_LOG2E;
    __declspec(target(MIC) align(4096)) static const real  VLOG2E = VOLATILITY*M_LOG2E;

    __declspec(target(MIC) align(4096)) real random [BLOCKSIZE];
    VSLStreamStatePtr Randomstream;
    vslNewStream(&Randomstream, VSL_BRNG_MT19937, RANDSEED);
#ifdef _OPENMP
//    kmp_set_defaults("KMP_AFFINITY=compact,granularity=fine");
#endif
#pragma omp parallel for
    for(int opt = 0; opt < OPT_N; opt++)
    {
        h_CallResult[opt]     = 0.0f;
        h_CallConfidence[opt] = 0.0f;
    }

    const int nblocks = RAND_N/BLOCKSIZE;
    for(int block = 0; block < nblocks; ++block)
    {

         getRngGaussian(VSL_METHOD_SGAUSSIAN_ICDF, Randomstream, BLOCKSIZE, random, (real)0.0f, (real)1.0f);
//        vsRngGaussian (VSL_METHOD_SGAUSSIAN_ICDF, Randomstream, BLOCKSIZE, random, 0.0f, 1.0f);
#pragma omp parallel for
    for(int opt = 0; opt < OPT_N; opt++)
    {
        real VBySqrtT = VLOG2E * sqrtf(T[opt]);
        real MuByT = RVVLOG2E * T[opt];
        real Sval = S[opt];
        real Xval = X[opt];
        real val = 0.0, val2 = 0.0;

#pragma vector aligned
#pragma simd reduction(+:val) reduction(+:val2)
#pragma unroll(4)
        for(int pos = 0; pos < BLOCKSIZE; pos++)
        {
            real callValue  = Sval * exp2f(MuByT + VBySqrtT * random[pos]) - Xval;
            callValue = (callValue > 0) ? callValue : 0;
            val  += callValue;
            val2 += callValue * callValue;
        }

            h_CallResult[opt] +=  val;
            h_CallConfidence[opt] +=  val2;
    }
    }
#pragma omp parallel for
    for(int opt = 0; opt < OPT_N; opt++)
    {
        const real val      = h_CallResult[opt];
        const real val2     = h_CallConfidence[opt];
        const real  exprt    = exp2f(-RLOG2E*T[opt]);
        h_CallResult[opt]     = exprt * val * INV_RAND_N;
        const real  stdDev   = sqrtf((F_RAND_N * val2 - val * val) * STDDEV_DENOM);
        h_CallConfidence[opt] = (real)(exprt * stdDev * CONFIDENCE_DENOM);
    }
    vslDeleteStream(&Randomstream);
}

#pragma offload_attribute(pop)

