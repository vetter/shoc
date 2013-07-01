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

// Rob Van der Wijngaart (Intel Corp) created C version
// V1.3 Nathan Wichmann made further improvements
// V1.2 Ramanan fixed few bugs Apr08
// V1.1 John Levesque vectorized the code Feb08
// V1.0 was from Sandia

//     March 09, 2007
//     
//     18-step reduced mechanism for ehtylene-air
//
//     By Tianfeng Lu
//     Princeton University
//     Email: tlu@princeton.edu

#include "S3D.h"

template <class real, int MAXVL>
__declspec(target(mic)) void 
getrates_i_VEC(real *P, real *T, real *Y, int *ICKWRK, real *RCKWRK, real *WDOT)
{

    const int IREAC=206, KK=22, KSS=10, KTOTAL=32;
    ALIGN64 real C[MAXVL*22], RF[MAXVL*IREAC], RB[MAXVL*IREAC], RKLOW[MAXVL*21],
        XQ[MAXVL*KSS], TI[MAXVL], TI2[MAXVL], SUM[MAXVL];
    const real SMALL = floatMin<real>(); //1.e-50;
    int VL = MAXVL;
    int I, K;

    for (I=1; I<=VL; I++) 
    {
        C(I,1) = Y(I,1)*4.96046521e-1;
        C(I,2) = Y(I,2)*9.92093043e-1;
        C(I,3) = Y(I,3)*6.25023433e-2;
        C(I,4) = Y(I,4)*3.12511716e-2;
        C(I,5) = Y(I,5)*5.87980383e-2;
        C(I,6) = Y(I,6)*5.55082499e-2;
        C(I,7) = Y(I,7)*3.02968146e-2;
        C(I,8) = Y(I,8)*2.93990192e-2;
        C(I,9) = Y(I,9)*6.65112065e-2;
        C(I,10) = Y(I,10)*6.23323639e-2;
        C(I,11) = Y(I,11)*3.57008335e-2;
        C(I,12) = Y(I,12)*2.27221341e-2;
        C(I,13) = Y(I,13)*3.33039255e-2;
        C(I,14) = Y(I,14)*3.84050525e-2;
        C(I,15) = Y(I,15)*3.56453112e-2;
        C(I,16) = Y(I,16)*3.32556033e-2;
        C(I,17) = Y(I,17)*2.4372606e-2;
        C(I,18) = Y(I,18)*2.37882046e-2;
        C(I,19) = Y(I,19)*2.26996304e-2;
        C(I,20) = Y(I,20)*2.43467162e-2;
        C(I,21) = Y(I,21)*2.37635408e-2;
        C(I,22) = Y(I,22)*3.56972032e-2;
    }

    for (I = 1; I <= VL; I++) SUM(I) = 0.0;
    for (K = 1; K <= 22; K++) 
    {
        for (I=1; I<=VL; I++) 
        {
            SUM(I) = SUM(I) + C(I,K);
        }
    }

    for (K = 1; K <= VL; K++) SUM(K) = P(K)/(SUM(K)*T(K)*8.314510e7);

    for (K = 1; K <= 22; K++) 
    {
        for (I=1; I<=VL; I++) 
        {
            C(I,K) = MAX(C(I,K), SMALL) * SUM(I);
        }
    }

    ratt_i_VEC<real,MAXVL>(T, RF, RB, RKLOW);
    ratx_i_VEC<real,MAXVL>(T, C, RF, RB, RKLOW);
    qssa_i_VEC<real,MAXVL>(RF, RB, XQ);
    rdwdot_i_VEC<real,MAXVL>(RF, RB, WDOT);
}

template <class real, int MAXVL>
__declspec(target(mic)) void 
getrates_i_(real *P, real *T, real *Y, int *VLp, int *ICKWRK, 
        real *RCKWRK, real *WDOT) 
{

    const int IREAC=206, KK=22, KSS=10, KTOTAL=32;
    ALIGN64 real C[MAXVL*22], RF[MAXVL*IREAC], RB[MAXVL*IREAC], RKLOW[MAXVL*21],
                 XQ[MAXVL*KSS], TI[MAXVL], TI2[MAXVL], SUM[MAXVL];
    const real SMALL = floatMin<real>() ; //1.e-50;
    int VL = *VLp;
    int I, K;

    for (I=1; I<=VL; I++) 
    {
        C(I,1) = Y(I,1)*4.96046521e-1;
        C(I,2) = Y(I,2)*9.92093043e-1;
        C(I,3) = Y(I,3)*6.25023433e-2;
        C(I,4) = Y(I,4)*3.12511716e-2;
        C(I,5) = Y(I,5)*5.87980383e-2;
        C(I,6) = Y(I,6)*5.55082499e-2;
        C(I,7) = Y(I,7)*3.02968146e-2;
        C(I,8) = Y(I,8)*2.93990192e-2;
        C(I,9) = Y(I,9)*6.65112065e-2;
        C(I,10) = Y(I,10)*6.23323639e-2;
        C(I,11) = Y(I,11)*3.57008335e-2;
        C(I,12) = Y(I,12)*2.27221341e-2;
        C(I,13) = Y(I,13)*3.33039255e-2;
        C(I,14) = Y(I,14)*3.84050525e-2;
        C(I,15) = Y(I,15)*3.56453112e-2;
        C(I,16) = Y(I,16)*3.32556033e-2;
        C(I,17) = Y(I,17)*2.4372606e-2;
        C(I,18) = Y(I,18)*2.37882046e-2;
        C(I,19) = Y(I,19)*2.26996304e-2;
        C(I,20) = Y(I,20)*2.43467162e-2;
        C(I,21) = Y(I,21)*2.37635408e-2;
        C(I,22) = Y(I,22)*3.56972032e-2;
    }

    for (I = 1; I <= VL; I++) SUM(I) = 0.0;
    for (K = 1; K <= 22; K++) {
        for (I=1; I<=VL; I++) {
            SUM(I) = SUM(I) + C(I,K);
        }
    }

    for (K = 1; K <= VL; K++) SUM(K) = P(K)/(SUM(K)*T(K)*8.314510e7);

    for (K = 1; K <= 22; K++) {
        for (I=1; I<=VL; I++) {
            C(I,K) = MAX(C(I,K), SMALL) * SUM(I);
        }
    }

    ratt_i_<real,MAXVL>(VLp,T, RF, RB, RKLOW);
    ratx_i_<real,MAXVL>(VLp,T, C, RF, RB, RKLOW);
    qssa_i_<real,MAXVL>(VLp,RF, RB, XQ);
    rdwdot_i_<real,MAXVL>(VLp,RF, RB, WDOT);
}

