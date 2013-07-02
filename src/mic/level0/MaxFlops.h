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

/****************************************************************************
 * Intel MIC(Many Integrated Core) version of SHOC MaxFlops program
 *
 * Authors: Zhi Ying (zhi.ying@intel.com)
 *          Jun Jin(jun.i.jin@intel.com)
 *
 * Creation: May 16, 2011
 *
 * Modifications:
 * Dec. 12, 2012 - Kyle Spafford - Comments and SHOC code style.
 *    
 *****************************************************************************/
#ifndef _MAXFLOPS_H_
#define _MAXFLOPS_H_

#ifdef __MIC__
#pragma offload_attribute (push, target(mic))
#include <micvec.h>
#pragma offload_attribute (pop)
#endif

// The following macros are used to construct MaxFlops functions, they use the
// same operations from the CUDA and OpenCL versions
#define ADD1_OP   s=v-s;
#define ADD2_OP   ADD1_OP s2=v-s2; 
#define ADD4_OP   ADD2_OP s3=v-s3; s4=v-s4;
#define ADD8_OP   ADD4_OP s5=v-s5; s6=v-s6; s7=v-s7; s8=v-s8;

#define MUL1_OP   s=s*s*v;
#define MUL2_OP   MUL1_OP s2=s2*s2*v;
#define MUL4_OP   MUL2_OP s3=s3*s3*v; s4=s4*s4*v;
#define MUL8_OP   MUL4_OP s5=s5*s5*v; s6=s6*s6*v; s7=s7*s7*v; s8=s8*s8*v;

#define MADD1_OP  s=v1-s*v2;
#define MADD2_OP  MADD1_OP s2=v1-s2*v2;
#define MADD4_OP  MADD2_OP s3=v1-s3*v2; s4=v1-s4*v2;
#define MADD8_OP  MADD4_OP s5=v1-s5*v2; s6=v1-s6*v2; s7=v1-s7*v2; s8=v1-s8*v2;

#define MULMADD1_OP  s=(v1-v2*s)*s;
#define MULMADD2_OP  MULMADD1_OP s2=(v1-v2*s2)*s2;
#define MULMADD4_OP  MULMADD2_OP s3=(v1-v2*s3)*s3; s4=(v1-v2*s4)*s4;
#define MULMADD8_OP  MULMADD4_OP s5=(v1-v2*s5)*s5; s6=(v1-v2*s6)*s6; \
                                 s7=(v1-v2*s7)*s7; s8=(v1-v2*s8)*s8;

#define ADD1_MOP20  \
     ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP \
     ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP \
     ADD1_OP ADD1_OP
#define ADD2_MOP20  \
     ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP \
     ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP \
     ADD2_OP ADD2_OP
#define ADD4_MOP10  \
     ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP \
     ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP
#define ADD8_MOP5  \
     ADD8_OP ADD8_OP ADD8_OP ADD8_OP ADD8_OP

#define MUL1_MOP20  \
     MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP \
     MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP \
     MUL1_OP MUL1_OP
#define MUL2_MOP20  \
     MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP \
     MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP \
     MUL2_OP MUL2_OP
#define MUL4_MOP10  \
     MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP \
     MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP
#define MUL8_MOP5  \
     MUL8_OP MUL8_OP MUL8_OP MUL8_OP MUL8_OP

#define MADD1_MOP20  \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP
#define MADD2_MOP20  \
     MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP \
     MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP \
     MADD2_OP MADD2_OP MADD2_OP MADD2_OP
#define MADD4_MOP10  \
     MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP \
     MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP
#define MADD8_MOP5  \
     MADD8_OP MADD8_OP MADD8_OP MADD8_OP MADD8_OP

#define MULMADD1_MOP20  \
     MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP \
     MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP \
     MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP \
     MULMADD1_OP MULMADD1_OP
#define MULMADD2_MOP20  \
     MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP \
     MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP \
     MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP \
     MULMADD2_OP MULMADD2_OP
#define MULMADD4_MOP10  \
     MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP \
     MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP
#define MULMADD8_MOP5  \
     MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP

template <class T2>
__declspec(target(mic))
inline bool micDp(void);

template <> 
inline bool micDp<float>(void) { return false; }

template <>
inline bool micDp<double>(void) { return true; }
     
template <class T>
__declspec(target(mic)) void Add1(const int num, T *data, const int nIters, 
        const T v) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = data[gid];
        for (int j=0 ; j<nIters ; ++j) 
        {
            ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 
            ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20
        }
        data[gid] = s;
    }
}

template <class T>
__declspec(target(mic)) void Add2(const int num, T *data, const int nIters,
        const T v) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = data[gid];
        __declspec(target(mic)) register T s2 = (T)(10.0f)-s;
        for (int j=0 ; j<nIters ; ++j) 
        {
            ADD2_MOP20 ADD2_MOP20 ADD2_MOP20
            ADD2_MOP20 ADD2_MOP20 ADD2_MOP20
        }
        data[gid] = s+s2;
    }
}

template <class T>
__declspec(target(mic)) void Add4(const int num, T *data, const int nIters,
        const T v) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = data[gid];
        __declspec(target(mic)) register T s2 = (T)(10.0f)-s;
        __declspec(target(mic)) register T s3 = (T)(9.0f)-s;
        __declspec(target(mic)) register T s4 = (T)(9.0f)-s2;
        for (int j=0 ; j<nIters ; ++j) 
        {
            ADD4_MOP10 ADD4_MOP10 ADD4_MOP10
                ADD4_MOP10 ADD4_MOP10 ADD4_MOP10
        }
        data[gid] = (s+s2)+(s3+s4);
    }
}

template <class T>
__declspec(target(mic)) void Add8(const int num, T *data, const int nIters,
        const T v) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = data[gid];
        __declspec(target(mic)) register T s2 = (T)(10.0f)-s;
        __declspec(target(mic)) register T s3 = (T)(9.0f)-s;
        __declspec(target(mic)) register T s4 = (T)(9.0f)-s2;
        __declspec(target(mic)) register T s5 = (T)(8.0f)-s;
        __declspec(target(mic)) register T s6 = (T)(8.0f)-s2;
        __declspec(target(mic)) register T s7 = (T)(7.0f)-s;
        __declspec(target(mic)) register T s8 = (T)(7.0f)-s2;

        for (int j=0 ; j<nIters ; ++j) 
        {
            ADD8_MOP5 ADD8_MOP5 ADD8_MOP5
            ADD8_MOP5 ADD8_MOP5 ADD8_MOP5
        }
        data[gid] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
    }
}

template <class T>
__declspec(target(mic)) void Mul1(const int num, T *data, const int nIters,
        const T v) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = (T)(0.999f);
        for (int j=0; j<nIters; ++j) 
        {
            MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
            MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
        }
        data[gid] = s;
    }
}

template <class T>
__declspec(target(mic)) void Mul2(const int num, T *data, const int nIters,
        const T v) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s  =   (T)0.999f;
        __declspec(target(mic)) register T s2 = s-(T)0.0001f;
        for (int j=0; j<nIters ; ++j) 
        {
            MUL2_MOP20 MUL2_MOP20 MUL2_MOP20 MUL2_MOP20 MUL2_MOP20
        }
        data[gid] = s+s2;
    }
}

template <class T>
__declspec(target(mic)) void Mul4(const int num, T *data, const int nIters,
        const T v) {
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s  =   (T)0.999f;
        __declspec(target(mic)) register T s2 = s-(T)0.0001f;
        __declspec(target(mic)) register T s3 = s-(T)0.0002f;
        __declspec(target(mic)) register T s4 = s-(T)0.0003f;
        for (int j=0; j<nIters; ++j) 
        {
             MUL4_MOP10 MUL4_MOP10 MUL4_MOP10 MUL4_MOP10 MUL4_MOP10
        }
        data[gid] = (s+s2)+(s3+s4);
    }
}

template <class T>
__declspec(target(mic)) void Mul8(const int num, T *data, const int nIters,
        const T v) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s  =   (T)0.999f;
        __declspec(target(mic)) register T s2 = s-(T)0.0001f;
        __declspec(target(mic)) register T s3 = s-(T)0.0002f;
        __declspec(target(mic)) register T s4 = s-(T)0.0003f;
        __declspec(target(mic)) register T s5 = s-(T)0.0004f;
        __declspec(target(mic)) register T s6 = s-(T)0.0005f;
        __declspec(target(mic)) register T s7 = s-(T)0.0006f;
        __declspec(target(mic)) register T s8 = s-(T)0.0007f;
        for (int j=0 ; j<nIters ; ++j) 
        {
            MUL8_MOP5 MUL8_MOP5 MUL8_MOP5 MUL8_MOP5 MUL8_MOP5
        }
        data[gid] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
    }
}

template <class T>
__declspec(target(mic)) void MAdd1(const int num, T *data, const int nIters,
        const T v1, const T v2) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = data[gid];
        for (int j=0 ; j<nIters ; ++j) 
        {
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 
            MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 
            MADD1_MOP20 MADD1_MOP20
        }
        data[gid] = s;
    }
}

template <class T>
__declspec(target(mic)) void MAdd2(const int num, T *data, const int nIters,
        const T v1, const T v2) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s  = data[gid];
        __declspec(target(mic)) register T s2 = (T)(10.0f)-s;
        for (int j=0 ; j<nIters ; ++j) 
        {
            MADD2_MOP20 MADD2_MOP20 MADD2_MOP20 
            MADD2_MOP20 MADD2_MOP20 MADD2_MOP20
        }
        data[gid] = s+s2;
    }
}

template <class T>
__declspec(target(mic)) void MAdd4(const int num, T *data, const int nIters, 
        const T v1, const T v2) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = data[gid];
        __declspec(target(mic)) register T s2 = (T)(10.0f)-s;
        __declspec(target(mic)) register T s3 = (T)(9.0f)-s;
        __declspec(target(mic)) register T s4 = (T)(9.0f)-s2;
        for (int j=0 ; j<nIters ; ++j) 
        {
            MADD4_MOP10 MADD4_MOP10 MADD4_MOP10
            MADD4_MOP10 MADD4_MOP10 MADD4_MOP10
        }
        data[gid] = (s+s2)+(s3+s4);
    }
}

template <class T>
__declspec(target(mic)) void MAdd8(const int num, T *data, const int nIters, 
        const T v1, const T v2) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = data[gid];
        __declspec(target(mic)) register T s2 = (T)(10.0f)-s;
        __declspec(target(mic)) register T s3 = (T)(9.0f)-s;
        __declspec(target(mic)) register T s4 = (T)(9.0f)-s2;
        __declspec(target(mic)) register T s5 = (T)(8.0f)-s;
        __declspec(target(mic)) register T s6 = (T)(8.0f)-s2;
        __declspec(target(mic)) register T s7 = (T)(7.0f)-s;
        __declspec(target(mic)) register T s8 = (T)(7.0f)-s2;
        for (int j=0 ; j<nIters ; ++j) 
        {
            MADD8_MOP5 MADD8_MOP5 MADD8_MOP5
            MADD8_MOP5 MADD8_MOP5 MADD8_MOP5
        }
        data[gid] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
    }
}

template <class T>
__declspec(target(mic)) void MulMAdd1(const int num, T *data, 
        const int nIters, const T v1, const T v2) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = data[gid];
        for (int j=0 ; j<nIters ; ++j) 
        {
            MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
            MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
        }
        data[gid] = s;
    }
}

template <class T>
__declspec(target(mic)) void MulMAdd2(const int num, T *data, const int nIters,
        const T v1, const T v2) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = data[gid];
        __declspec(target(mic)) register T s2 = (T)(10.0f)-s;
        for (int j=0 ; j<nIters ; ++j) 
        {
            MULMADD2_MOP20 MULMADD2_MOP20
            MULMADD2_MOP20 MULMADD2_MOP20
        }
        data[gid] = s+s2;
    }
}

template <class T>
__declspec(target(mic)) void MulMAdd4(const int num, T *data, const int nIters,
        const T v1, const T v2) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = data[gid];
        __declspec(target(mic)) register T s2 = (T)(10.0f)-s;
        __declspec(target(mic)) register T s3 = (T)(9.0f)-s;
        __declspec(target(mic)) register T s4 = (T)(9.0f)-s2;
        for (int j=0 ; j<nIters ; ++j) 
        {
            MULMADD4_MOP10 MULMADD4_MOP10
            MULMADD4_MOP10 MULMADD4_MOP10
        }
        data[gid] = (s+s2)+(s3+s4);
    }
}

template <class T>
__declspec(target(mic)) void MulMAdd8(const int num, T *data, const int nIters,
        const T v1, const T v2) 
{
    #pragma omp parallel for 
    for (int gid = 0; gid<num; gid++)
    {
        __declspec(target(mic)) register T s = data[gid];
        __declspec(target(mic)) register T s2 = (T)(10.0f)-s;
        __declspec(target(mic)) register T s3 = (T)(9.0f)-s;
        __declspec(target(mic)) register T s4 = (T)(9.0f)-s2;
        __declspec(target(mic)) register T s5 = (T)(8.0f)-s;
        __declspec(target(mic)) register T s6 = (T)(8.0f)-s2;
        __declspec(target(mic)) register T s7 = (T)(7.0f)-s;
        __declspec(target(mic)) register T s8 = (T)(7.0f)-s2;
        for (int j=0 ; j<nIters ; ++j) 
        {
            MULMADD8_MOP5 MULMADD8_MOP5
            MULMADD8_MOP5 MULMADD8_MOP5
        }
        data[gid] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
    }
}

// Vector versions of functions to take advantage of SIMD

template <class T>
__declspec(target(mic)) void Add1_MIC(const int num, T *data, const int nIters,
        const T v) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        Add1(num/8, (F64vec8 *)data, nIters, (F64vec8)v);
    }
    else
    {
        Add1(num/16, (F32vec16 *)data, nIters, (F32vec16)v);
    }
#else
    Add1(num, data, nIters, v);
#endif
}

template <class T>
__declspec(target(mic)) void Add2_MIC(const int num, T *data, const int nIters,
        const T v) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        Add2(num/8, (F64vec8 *)data, nIters, (F64vec8)v);
    }
    else
    {
        Add2(num/16, (F32vec16 *)data, nIters, (F32vec16)v);
    }
#else
    Add2(num, data, nIters, v);
#endif
}

template <class T>
__declspec(target(mic)) void Add4_MIC(const int num, T *data, const int nIters,
        const T v) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        Add4(num/8, (F64vec8 *)data, nIters, (F64vec8)v);
    }
    else
    {
        Add4(num/16, (F32vec16 *)data, nIters, (F32vec16)v);
    }
#else
    Add4(num, data, nIters, v);
#endif
}

template <class T>
__declspec(target(mic)) void Add8_MIC(const int num, T *data, const int nIters,
        const T v) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        Add8(num/8, (F64vec8 *)data, nIters, (F64vec8)v);
    }
    else
    {
        Add8(num/16, (F32vec16 *)data, nIters, (F32vec16)v);
    }
#else
    Add8(num, data, nIters, v);
#endif
}

template <class T>
__declspec(target(mic)) void Mul1_MIC(const int num, T *data, const int nIters,
        const T v) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        Mul1(num/8, (F64vec8 *)data, nIters, (F64vec8)v);
    }
    else
    {
        Mul1(num/16, (F32vec16 *)data, nIters, (F32vec16)v);
    }
#else
    Mul1(num, data, nIters, v);
#endif
}

template <class T>
__declspec(target(mic)) void Mul2_MIC(const int num, T *data, const int nIters,
        const T v) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        Mul2(num/8, (F64vec8 *)data, nIters, (F64vec8)v);
    }
    else
    {
        Mul2(num/16, (F32vec16 *)data, nIters, (F32vec16)v);
    }
#else
    Mul2(num, data, nIters, v);
#endif
}

template <class T>
__declspec(target(mic)) void Mul4_MIC(const int num, T *data, const int nIters,
        const T v) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        Mul4(num/8, (F64vec8 *)data, nIters, (F64vec8)v);
    }
    else
    {
        Mul4(num/16, (F32vec16 *)data, nIters, (F32vec16)v);
    }
#else
    Mul4(num, data, nIters, v);
#endif
}

template <class T>
__declspec(target(mic)) void Mul8_MIC(const int num, T *data, const int nIters,
        const T v) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        Mul8(num/8, (F64vec8 *)data, nIters, (F64vec8)v);
    }
    else
    {
        Mul8(num/16, (F32vec16 *)data, nIters, (F32vec16)v);
    }
#else
    Mul8(num, data, nIters, v);
#endif
}

template <class T>
__declspec(target(mic)) void MAdd1_MIC(const int num, T *data, const int nIters,
        const T v1, const T v2) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        MAdd1(num/8, (F64vec8 *)data, nIters, (F64vec8)v1, (F64vec8)v2);
    }
    else
    {
        MAdd1(num/16, (F32vec16 *)data, nIters, (F32vec16)v1, (F32vec16)v2);
    }
#else
    MAdd1(num, data, nIters, v1, v2);
#endif
}

template <class T>
__declspec(target(mic)) void MAdd2_MIC(const int num, T *data, const int nIters,
        const T v1, const T v2) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        MAdd2(num/8, (F64vec8 *)data, nIters, (F64vec8)v1, (F64vec8)v2);
    }
    else
    {
        MAdd2(num/16, (F32vec16 *)data, nIters, (F32vec16)v1, (F32vec16)v2);
    }
#else
    MAdd2(num, data, nIters, v1, v2);
#endif
}

template <class T>
__declspec(target(mic)) void MAdd4_MIC(const int num, T *data, const int nIters,
        const T v1, const T v2) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        MAdd4(num/8, (F64vec8 *)data, nIters, (F64vec8)v1, (F64vec8)v2);
    }
    else
    {
        MAdd4(num/16, (F32vec16 *)data, nIters, (F32vec16)v1, (F32vec16)v2);
    }
#else
    MAdd4(num, data, nIters, v1, v2);
#endif
}

template <class T>
__declspec(target(mic)) void MAdd8_MIC(const int num, T *data, const int nIters,
        const T v1, const T v2) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        MAdd8(num/8, (F64vec8 *)data, nIters, (F64vec8)v1, (F64vec8)v2);
    }
    else
    {
        MAdd8(num/16, (F32vec16 *)data, nIters, (F32vec16)v1, (F32vec16)v2);
    }
#else
    MAdd8(num, data, nIters, v1, v2);
#endif
}

template <class T>
__declspec(target(mic)) void MulMAdd1_MIC(const int num, T *data, 
        const int nIters, const T v1, const T v2) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        MulMAdd1(num/8, (F64vec8 *)data, nIters, (F64vec8)v1, (F64vec8)v2);
    }
    else
    {
        MulMAdd1(num/16, (F32vec16 *)data, nIters, (F32vec16)v1, (F32vec16)v2);
    }
#else
    MulMAdd1(num, data, nIters, v1, v2);
#endif
}

template <class T>
__declspec(target(mic)) void MulMAdd2_MIC(const int num, T *data, 
        const int nIters, const T v1, const T v2) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        MulMAdd2(num/8, (F64vec8 *)data, nIters, (F64vec8)v1, (F64vec8)v2);
    }
    else
    {
        MulMAdd2(num/16, (F32vec16 *)data, nIters, (F32vec16)v1, (F32vec16)v2);
    }
#else
    MulMAdd1(num, data, nIters, v1, v2);
#endif
}

template <class T>
__declspec(target(mic)) void MulMAdd4_MIC(const int num, T *data, 
        const int nIters, const T v1, const T v2) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        MulMAdd4(num/8, (F64vec8 *)data, nIters, (F64vec8)v1, (F64vec8)v2);
    }
    else
    {
        MulMAdd4(num/16, (F32vec16 *)data, nIters, (F32vec16)v1, (F32vec16)v2);
    }
#else
    MulMAdd4(num, data, nIters, v1, v2);
#endif
}

template <class T>
__declspec(target(mic)) void MulMAdd8_MIC(const int num, T *data, 
        const int nIters, const T v1, const T v2) 
{
#ifdef __MIC__    
    if(micDp<T>())
    {
        MulMAdd8(num/8, (F64vec8 *)data, nIters, (F64vec8)v1, (F64vec8)v2);
    }
    else
    {
        MulMAdd8(num/16, (F32vec16 *)data, nIters, (F32vec16)v1, (F32vec16)v2);
    }
#else
    MulMAdd8(num, data, nIters, v1, v2);
#endif
}

#endif // _MAX_FLOPS_H_
