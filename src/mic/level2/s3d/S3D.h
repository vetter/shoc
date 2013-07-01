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

#ifndef GPU_GLOBAL_H
#define GPU_GLOBAL_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "offload.h"
#include <math.h>
#include "omp.h"
#include "floatmin.h"
#define ALIGN (4096)

#define RESTRICT __restrict__

// Replace divisions by multiplication with the reciprocal
#define REPLACE_DIV_WITH_RCP 1

#if REPLACE_DIV_WITH_RCP
template <class T1, class T2>
__declspec(target(mic))  __attribute__((vector)) T1 DIV(T1 x, T2 y)
{
   return x * (1.0f / y);
}
#else
template <class T1, class T2>
__declspec(target(mic))  __attribute__((vector)) T1 DIV(T1 x, T2 y)
{
   return x / y;
}
#endif

// Choose correct intrinsics based on precision
// POW
template<class T>
__declspec(target(mic))  __attribute__((vector)) T POW (T in, T in2);

#pragma offload_attribute(push,target(mic))
template<>
 __attribute__((vector)) double POW<double>(double in, double in2)
{
    return pow(in, in2);
}

template<>
 __attribute__((vector))  float POW<float>(float in, float in2)
{
    return powf(in, in2);
}
// EXP
template<class T>
__declspec(target(mic)) __attribute__((vector)) T EXP(T in);
//__declspec(target(mic)) T EXP(T in);

template<>
 __attribute__((vector)) double EXP<double>(double in)
{
    return exp(in);
}

template<>
__attribute__((vector)) float EXP<float>(float in)
{
    return expf(in);
}

// EXP10
template<class T>
__declspec(target(mic))  __attribute__((vector))  T EXP10(T in);

template<>
 __attribute__((vector))  double EXP10<double>(double in)
{
    return exp10(in);
}

template<>
 __attribute__((vector))  float EXP10<float>(float in)
{
    return exp10f(in);
}

// EXP2
template<class T>
 __attribute__((vector))  T EXP2(T in);

template<>
 __attribute__((vector))  double EXP2<double>(double in)
{
    return exp2(in);
}

template<>
 __attribute__((vector))  float EXP2<float>(float in)
{
    return exp2f(in);
}

#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))


// LOG
template<class T>
__declspec(target(mic))  __attribute__((vector)) T LOG(T in);

template<>
__attribute__((vector)) double LOG<double>(double in)
{
    return log(in);
}

template<>
__attribute__((vector)) float LOG<float>(float in)
{
    return logf(in);
}

// LOG10
template<class T>
 __attribute__((vector))  T LOG10(T in);

template<>
 __attribute__((vector)) double LOG10<double>(double in)
{
    return log10(in);
}

template<>
 __attribute__((vector))  float LOG10<float>(float in)
{
    return log10f(in);
}

#pragma offload_attribute(pop)

// Size macros
// This is the number of floats/doubles per thread for each var

#define A_DIM                (11)
#define C_SIZE               (22)
#define RF_SIZE             (206)
#define RB_SIZE             (206)
#define WDOT_SIZE            (22)
#define RKLOW_SIZE           (21)
#define Y_SIZE               (22)
#define A_SIZE    (A_DIM * A_DIM)
#define EG_SIZE              (32)

// Rob port Macros
#define oneDarr(name,i) (name)[i-1]
#define vecarr(name,i,j) (name)[i-1+(MAXVL)*(j-1)]

#define A(i) oneDarr(A,i)
#define B(i) oneDarr(B,i)

#define ALIGN64 __declspec(align(64))

#define vrda_exp_(countp, arr1, arr2)                                   \
  for (I=0; I<(*(countp)); I++) (*((arr2)+I)) = (EXP(*((arr1)+I)))

#define vrda_log_(countp, arr1, arr2)                                   \
  for (I=0; I<(*(countp)); I++) (*((arr2)+I)) = (LOG(*((arr1)+I)))

#define vrda_log10_(countp, arr1, arr2)                                 \
  for (I=0; I<(*(countp)); I++) (*((arr2)+I)) = (LOG10(*((arr1)+I)))

#define RECIP 0.43429448190325176116


#define P(i)        oneDarr(P,i)
#define ICKWRK(i)   oneDarr(ICKWRK,i)
#define RCKWRK(i)   oneDarr(RCKWRK,i)
#define TI(i)       oneDarr(TI,i)
#define TI2(i)      oneDarr(TI2,i)
#define T(i)        oneDarr(T,i)
#define SUM(i)      oneDarr(SUM,i)
#define Y(i,j)      vecarr(Y,i,j)
#define WDOT(i,j)   vecarr(WDOT,i,j)
#define RKF(i,j)   vecarr(RKF,i,j)
#define RKR(i,j)   vecarr(RKR,i,j)
#define ROP(i,j)   vecarr(ROP,i,j)
#define RKLOW(i,j)  vecarr(RKLOW,i,j)
#define RB(i,j)     vecarr(RB,i,j)
#define RF(i,j)     vecarr(RF,i,j)
#define C(i,j)      vecarr(C,i,j)
#define CTB(i,j)    vecarr(CTB,i,j)
#define CTOT(i)     oneDarr(CTOT,i)
#define PR(i)       oneDarr(PR,i)
#define PCOR(i)     oneDarr(PCOR,i)
#define PRLOG(i)    oneDarr(PRLOG,i)
#define FCENT0(i)   oneDarr(FCENT0,i)
#define FCENT1(i)   oneDarr(FCENT1,i)
#define FCENT2(i)   oneDarr(FCENT2,i)
#define FCLOG(i)    oneDarr(FCLOG,i)
#define XN(i)       oneDarr(XN,i)
#define CPRLOG(i)   oneDarr(CPRLOG,i)
#define FLOG(i)     oneDarr(FLOG,i)
#define FC(i)       oneDarr(FC,i)
#define FCENT(i)    oneDarr(FCENT,i)
#define XQ(i,j)     vecarr(XQ,i,j)

#define SMH(i,j)   vecarr(SMH,i,j)
#define EG(i,j)    vecarr(EG,i,j)
#define EGI(i,j)   vecarr(EGI,i,j)
#define EQK(i,j)   vecarr(EQK,i,j)
#define ALOGT(i)   oneDarr(ALOGT,i)
#define TMP(i)     oneDarr(TMP,i)
#define vecof10(i)  oneDarr(vecof10,i)
#define TN(i,j)   vecarr(TN,i,j)
#define TLOG(i)   oneDarr(TLOG,i)

#endif
