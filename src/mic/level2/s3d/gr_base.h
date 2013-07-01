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


#ifndef GETRATES_BASE_H
#define GETRATES_BASE_H

#include "S3D.h"

template <class real>
__declspec(target(mic)) void
gr_base(const real* P, const real* T, const real* Y, real* C, real TCONV,
        real PCONV, const int n) 
{
    #pragma ivdep
    #pragma vector always
    #pragma omp parallel for
    for(int el=0;el<n;el++)
    {

        const real TEMP = T[el]*TCONV;
        const real PRES = P[el]*PCONV;
        const real SMALL = floatMin<real>();//FLT_MIN;

        real SUM, ctmp;

        SUM = 0.0f;

        C(1)  = ctmp = Y(1) *4.96046521e-1;
        SUM  += ctmp;
        C(2)  = ctmp = Y(2) *9.92093043e-1;
        SUM  += ctmp;
        C(3)  = ctmp = Y(3) *6.25023433e-2;
        SUM  += ctmp;
        C(4)  = ctmp = Y(4) *3.12511716e-2;
        SUM  += ctmp;
        C(5)  = ctmp = Y(5) *5.87980383e-2;
        SUM  += ctmp;
        C(6)  = ctmp = Y(6) *5.55082499e-2;
        SUM  += ctmp;
        C(7)  = ctmp = Y(7) *3.02968146e-2;
        SUM  += ctmp;
        C(8)  = ctmp = Y(8) *2.93990192e-2;
        SUM  += ctmp;
        C(9)  = ctmp = Y(9) *6.65112065e-2;
        SUM  += ctmp;
        C(10) = ctmp = Y(10)*6.23323639e-2;
        SUM  += ctmp;
        C(11) = ctmp = Y(11)*3.57008335e-2;
        SUM  += ctmp;
        C(12) = ctmp = Y(12)*2.27221341e-2;
        SUM  += ctmp;
        C(13) = ctmp = Y(13)*3.33039255e-2;
        SUM  += ctmp;
        C(14) = ctmp = Y(14)*3.84050525e-2;
        SUM  += ctmp;
        C(15) = ctmp = Y(15)*3.56453112e-2;
        SUM  += ctmp;
        C(16) = ctmp = Y(16)*3.32556033e-2;
        SUM  += ctmp;
        C(17) = ctmp = Y(17)*2.4372606e-2;
        SUM  += ctmp;
        C(18) = ctmp = Y(18)*2.37882046e-2;
        SUM  += ctmp;
        C(19) = ctmp = Y(19)*2.26996304e-2;
        SUM  += ctmp;
        C(20) = ctmp = Y(20)*2.43467162e-2;
        SUM  += ctmp;
        C(21) = ctmp = Y(21)*2.37635408e-2;
        SUM  += ctmp;
        C(22) = ctmp = Y(22)*3.56972032e-2;
        SUM  += ctmp;
        SUM = DIV (PRES, (SUM * (TEMP) * 8.314510e7));

        #pragma unroll 22
        for (unsigned k=1; k<=22; k++) 
        {
            C(k) = MAX(C(k), SMALL) * SUM;
        }
    }
}

#endif
