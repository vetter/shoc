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

#ifndef FFTLIB_H
#define FFTLIB_H

#include <omp.h>
#include <math.h>

#pragma offload_attribute(push, target(mic))
#include <mkl.h>
#include <mkl_dfti.h>
#pragma offload_attribute(pop)

struct cplxflt {
    float x;
    float y;
};

struct cplxdbl {
    double x;
    double y;
};

#pragma offload_attribute(push, target(mic))
template <class T2>
__declspec(target(mic))
inline bool micDp(void);
template <>
inline bool micDp<cplxflt>(void) { return false; }
template <>
inline bool micDp<cplxdbl>(void) { return true; }
template <class T2>
__declspec(target(mic))
void forward(T2* source, const int fftsz, const int n_ffts);
template <class T2>
__declspec(target(mic))
void inverse(T2* source, const int fftsz, const int n_ffts);
template <class T2>
__declspec(target(mic))
int checkDiff(T2 *source, const int half_n_cmplx);
#pragma offload_attribute(pop)

// Perform forward ffts
template<class T2>
__declspec(target(mic))
void forward(T2* source, const int fftsz, const int n_ffts)
{
    static __declspec(target(mic)) DFTI_DESCRIPTOR_HANDLE plan;
    if (!source)
    {
        if (fftsz <= 0)
        {
            DftiFreeDescriptor(&plan);
            return;
        }
        if (micDp<T2>())
        {
            DftiCreateDescriptor(&plan, DFTI_DOUBLE, DFTI_COMPLEX, 1, 
                    (MKL_LONG)fftsz);
        }
        else
        {
            DftiCreateDescriptor(&plan, DFTI_SINGLE, DFTI_COMPLEX, 1, 
                    (MKL_LONG)fftsz);
        }
        DftiSetValue(plan, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)n_ffts);
        DftiSetValue(plan, DFTI_INPUT_DISTANCE, (MKL_LONG)fftsz);
        DftiSetValue(plan, DFTI_OUTPUT_DISTANCE, (MKL_LONG)fftsz);
        DftiCommitDescriptor(plan);
    }
    DftiComputeForward(plan, source);
}

// Perform inverse ffts
template<class T2>
__declspec(target(mic))
void inverse(T2* source, const int fftsz, const int n_ffts)
{
    static __declspec(target(mic)) DFTI_DESCRIPTOR_HANDLE plan;
    if (!source)
    {
        if (fftsz <= 0)
        {
            DftiFreeDescriptor(&plan);
            return;
        }
        if (micDp<T2>())
        {
            DftiCreateDescriptor(&plan, DFTI_DOUBLE, DFTI_COMPLEX, 1, 
                    (MKL_LONG)fftsz);
        }
        else
        {
            DftiCreateDescriptor(&plan, DFTI_SINGLE, DFTI_COMPLEX, 1, 
                    (MKL_LONG)fftsz);
        }
        DftiSetValue(plan, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)n_ffts);
        DftiSetValue(plan, DFTI_INPUT_DISTANCE, (MKL_LONG)fftsz);
        DftiSetValue(plan, DFTI_OUTPUT_DISTANCE, (MKL_LONG)fftsz);
        //DftiSetValue(plan, DFTI_BACKWARD_SCALE, 1.0/fftsz);
        DftiCommitDescriptor(plan);
    }
    DftiComputeBackward(plan, source);
}
#endif
