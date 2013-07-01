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
//     18-step reduced mechanism for ehtylene-air
//
//     By Tianfeng Lu
//     Princeton University
//     Email: tlu@princeton.edu

#include "S3D.h"


template <class real, int MAXVL>
__declspec(target(mic)) void 
ratx_i_VEC(real * RESTRICT T, real * RESTRICT C, real * RESTRICT RF, 
        real * RESTRICT RB, real * RESTRICT RKLOW)
{

    ALIGN64        real TI[MAXVL], CTB[MAXVL*206], CTOT[MAXVL], PR[MAXVL],
                   PCOR[MAXVL], PRLOG[MAXVL], FCENT0[MAXVL], FCENT1[MAXVL], 
                   FCENT2[MAXVL], FCLOG[MAXVL], XN[MAXVL], CPRLOG[MAXVL],
                   FLOG[MAXVL],FC[MAXVL], FCENT[MAXVL], ALOGT[MAXVL], 
                   vecof10[MAXVL];

    int VL = MAXVL;
    int I, K;
    const real SMALL= floatMin<real>();//FLT_MIN ; //1.e-200;

    #pragma vector aligned
    for (I=1; I<=VL; I++) vecof10(I) =10.;
    #pragma vector aligned
    for (I=1; I<=VL; I++) TI(I)=1./T(I);

    vrda_log_(&VL, T, ALOGT);
    //
    //    third-body reactions
    //
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTOT(I) = 0.0;
    for (K = 1; K<=22; K++) {
    #pragma vector aligned
        for (I=1; I<=VL; I++) {
            CTOT(I) = CTOT(I) + C(I,K);
        }
    }

    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,5)  = CTOT(I) - C(I,1) - C(I,6) + C(I,10) - C(I,12) 
        + 2.e0*C(I,16) + 2.e0*C(I,14) + 2.e0*C(I,15);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,9)  = CTOT(I) - 2.7e-1*C(I,1) + 2.65e0*C(I,6) + C(I,10) 
        + 2.e0*C(I,16) + 2.e0*C(I,14) + 2.e0*C(I,15);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,10) = CTOT(I) + C(I,1) + 5.e0*C(I,6) + C(I,10)
        + 5.e-1*C(I,11) + C(I,12) 
            + 2.e0*C(I,16) + 2.e0*C(I,14) + 2.e0*C(I,15);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,31) = CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,39) = CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,41) = CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,46) = CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,48) = CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,56) = CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,71) = CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,78) = CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,89) = CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,93) = CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,115)= CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,126)= CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,132)= CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,145)= CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,148)= CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,155)= CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,156)= CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,170)= CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,185)= CTB(I,10);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,114)= CTOT(I)+C(I,1)+5.e0*C(I,6)+C(I,10) 
        +5.e-1*C(I,11)+C(I,12)  
            +2.e0*C(I,16)+1.5e0*C(I,14)+1.5e0*C(I,15);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,11) = CTOT(I)+1.4e0*C(I,1)+1.44e1*C(I,6)+C(I,10) 
        +7.5e-1*C(I,11)  
            +2.6e0*C(I,12)+2.e0*C(I,16)+2.e0*C(I,14)   
            +2.e0*C(I,15);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,12) = CTOT(I) - C(I,4) - C(I,6) - 2.5e-1*C(I,11) 
        +5.e-1*C(I,12)  
            +5.e-1*C(I,16) - C(I,22)+2.e0*C(I,14) 
            +2.e0*C(I,15);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,16) = CTOT(I)+C(I,1)+5.e0*C(I,6)+C(I,10)  
        +5.e-1*C(I,11) +C(I,12)  
            +2.e0*C(I,16)+2.e0*C(I,14)+2.e0*C(I,15);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,29) = CTOT(I)+C(I,1)+5.e0*C(I,4)+5.e0*C(I,6)+C(I,10)  
        +5.e-1*C(I,11)+2.5e0*C(I,12)+2.e0*C(I,16)  
            +2.e0*C(I,14)+2.e0*C(I,15);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,121)= CTOT(I); 
    #pragma vector aligned
    for (I=1; I<=VL; I++) CTB(I,190)= CTOT(I)+C(I,1)+5.e0*C(I,6)+C(I,10)
        +5.e-1*C(I,11)+C(I,12)+2.e0*C(I,16);

    //     If fall-off (pressure correction):
#define ALOG log

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,1) * CTB(I,16) / RF(I,16);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log(MAX(PR(I),SMALL));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/9.4e1);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.756e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.182e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 2.654e-1*FCENT0(I) +7.346e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,16) = RF(I,16) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,16) = RB(I,16) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,2) * CTB(I,31) / RF(I,31);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( MAX(PR(I),SMALL));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.97e2);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.54e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-1.03e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 6.8e-2*FCENT0(I) +9.32e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,31) = RF(I,31) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,31) = RB(I,31) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,3) * CTB(I,39) / RF(I,39);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/2.37e2);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.652e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.069e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 4.243e-1*FCENT0(I) +5.757e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( MAX(FCENT(I),SMALL));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,39) = RF(I,39) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,39) = RB(I,39) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,4) * CTB(I,41) / RF(I,41);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/2.71e2);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/2.755e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-6.57e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 2.176e-1*FCENT0(I) +7.824e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,41) = RF(I,41) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,41) = RB(I,41) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,5) * CTB(I,48) / RF(I,48);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/7.8e1);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.995e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.59e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 3.2e-1*FCENT0(I) +6.8e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log((MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,48) = RF(I,48) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,48) = RB(I,48) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,6) * CTB(I,56) / RF(I,56);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/2.75e2);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.226e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.185e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 4.093e-1*FCENT0(I) +5.907e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    //       vrda_exp_(VLp,FC,FC);;
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,56) = RF(I,56) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,56) = RB(I,56) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,7) * CTB(I,71) / RF(I,71);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/9.4e1);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.555e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-4.2e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 2.42e-1*FCENT0(I) +7.58e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( MAX(FCENT(I),SMALL));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,71) = RF(I,71) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,71) = RB(I,71) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,8) * CTB(I,78) / RF(I,78);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/7.4e1);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/2.941e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-6.964e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 2.17e-1*FCENT0(I) +7.83e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,78) = RF(I,78) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,78) = RB(I,78) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,9) * CTB(I,89) / RF(I,89);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.3076e1);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/2.078e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.093e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 3.827e-1*FCENT0(I) +6.173e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,89) = RF(I,89) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,89) = RB(I,89) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,10) * CTB(I,93) / RF(I,93);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.51e2);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.038e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-4.97e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 4.675e-1*FCENT0(I) +5.325e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,93) = RF(I,93) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,93) = RB(I,93) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,11) * CTB(I,114) / RF(I,114);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,114) = RF(I,114) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,114) = RB(I,114) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,12) * CTB(I,115) / RF(I,115);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/5.3837e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/4.2932e0);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(7.95e-2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = -9.816e-1*FCENT0(I) +1.9816e0*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,115) = RF(I,115) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,115) = RB(I,115) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,13) * CTB(I,126) / RF(I,126);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 6.63e-1*EXP(-T(I)/1.707e3) + 3.37e-1*EXP(-T(I)/3.2e3)
        + EXP(-4.131e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.707e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/3.2e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-4.131e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 6.63e-1*FCENT0(I) +3.37e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,126) = RF(I,126) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,126) = RB(I,126) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,14) * CTB(I,132) / RF(I,132);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/2.075e2);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/2.663e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-6.095e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 2.18e-1*FCENT0(I) +7.82e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,132) = RF(I,132) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,132) = RB(I,132) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,15) * CTB(I,145) / RF(I,145);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.3406e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/6.e4);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-1.01398e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 8.25e-1*FCENT0(I) + 1.75e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,145) = RF(I,145) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,145) = RB(I,145) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,16) * CTB(I,148) / RF(I,148);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/8.9e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/4.35e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-7.244e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 4.5e-1*FCENT0(I) + 5.5e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,148) = RF(I,148) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,148) = RB(I,148) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,17) * CTB(I,155) / RF(I,155);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.8e2);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.035e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.417e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 2.655e-1*FCENT0(I) + 7.345e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,155) = RF(I,155) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,155) = RB(I,155) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,18) * CTB(I,156) / RF(I,156);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( MAX(PR(I),SMALL));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/2.1e2);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/9.84e2);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-4.374e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 2.47e-2*FCENT0(I) + 9.753e-1*FCENT1(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,156) = RF(I,156) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,156) = RB(I,156) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,19) * CTB(I,170) / RF(I,170);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.25e2);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/2.219e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-6.882e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 1.578e-1*FCENT0(I) + 8.422e-1*FCENT1(I) + FCENT2(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,170) = RF(I,170) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,170) = RB(I,170) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,20) * CTB(I,185) / RF(I,185);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( MAX(PR(I),SMALL));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.0966e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-6.8595e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = 9.8e-1*FCENT0(I) + 2.e-2*FCENT0(I) + FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,185) = RF(I,185) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,185) = RB(I,185) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,21) * CTB(I,190) / RF(I,190);
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( MAX(PR(I),SMALL));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.31e3);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-4.8097e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCENT(I) = FCENT1(I)+FCENT2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( MAX(FCENT(I),SMALL));
    #pragma vector aligned
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    #pragma vector aligned
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,190) = RF(I,190) * PCOR(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RB(I,190) = RB(I,190) * PCOR(I);

    #pragma vector aligned
    for (I=1; I<=VL; I++) 
    {
        RF(I,1) = RF(I,1)*C(I,2)*C(I,4);
        RF(I,2) = RF(I,2)*C(I,3)*C(I,1);
        RF(I,3) = RF(I,3)*C(I,5)*C(I,1);
        RF(I,4) = RF(I,4)*C(I,5)*C(I,5);
        RF(I,5) = RF(I,5)*CTB(I,5)*C(I,2)*C(I,2);
        RF(I,6) = RF(I,6)*C(I,2)*C(I,2)*C(I,1);
        RF(I,7) = RF(I,7)*C(I,2)*C(I,2)*C(I,6);
        RF(I,8) = RF(I,8)*C(I,2)*C(I,2)*C(I,12);
        RF(I,9) = RF(I,9)*CTB(I,9)*C(I,2)*C(I,5);
        RF(I,10) = RF(I,10)*CTB(I,10)*C(I,3)*C(I,2);
        RF(I,11) = RF(I,11)*CTB(I,11)*C(I,3)*C(I,3);
        RF(I,12) = RF(I,12)*CTB(I,12)*C(I,2)*C(I,4);
        RF(I,13) = RF(I,13)*C(I,2)*C(I,4)*C(I,4);
        RF(I,14) = RF(I,14)*C(I,2)*C(I,4)*C(I,6);
        RF(I,15) = RF(I,15)*C(I,2)*C(I,4)*C(I,22);
        RF(I,16) = RF(I,16)*C(I,5)*C(I,5);
        RF(I,17) = RF(I,17)*C(I,7)*C(I,2);
        RF(I,18) = RF(I,18)*C(I,7)*C(I,2);
        RF(I,19) = RF(I,19)*C(I,7)*C(I,2);
        RF(I,20) = RF(I,20)*C(I,7)*C(I,3);
        RF(I,21) = RF(I,21)*C(I,7)*C(I,5);
        RF(I,22) = RF(I,22)*C(I,7)*C(I,7);
        RF(I,23) = RF(I,23)*C(I,7)*C(I,7);
        RF(I,24) = RF(I,24)*C(I,8)*C(I,2);
        RF(I,25) = RF(I,25)*C(I,8)*C(I,2);
        RF(I,26) = RF(I,26)*C(I,8)*C(I,3);
        RF(I,27) = RF(I,27)*C(I,8)*C(I,5);
        RF(I,28) = RF(I,28)*C(I,8)*C(I,5);
        RF(I,29) = RF(I,29)*CTB(I,29)*C(I,11)*C(I,3);
        RF(I,30) = RF(I,30)*C(I,11)*C(I,5);
        RF(I,31) = RF(I,31)*C(I,11)*C(I,1);
        RF(I,32) = RF(I,32)*C(I,11)*C(I,4);
        RF(I,33) = RF(I,33)*C(I,11)*C(I,7);
        RF(I,34) = RF(I,34)*C(I,3);
        RF(I,35) = RF(I,35)*C(I,5);
        RF(I,36) = RF(I,36)*C(I,1);
        RF(I,37) = RF(I,37)*C(I,6);
        RF(I,38) = RF(I,38)*C(I,4);
        RF(I,39) = RF(I,39)*C(I,11);
        RF(I,40) = RF(I,40)*C(I,12);
        RF(I,41) = RF(I,41)*C(I,2);
        RF(I,42) = RF(I,42)*C(I,2);
        RF(I,43) = RF(I,43)*C(I,3);
        RF(I,44) = RF(I,44)*C(I,3);
        RF(I,45) = RF(I,45)*C(I,5);
        RF(I,46) = RF(I,46)*CTB(I,46);
        RF(I,47) = RF(I,47)*C(I,4);
        RF(I,48) = RF(I,48)*C(I,2);
        RF(I,49) = RF(I,49)*C(I,1);
        RF(I,50) = RF(I,50)*C(I,3);
        RF(I,51) = RF(I,51)*C(I,4);
        RF(I,52) = RF(I,52)*C(I,4);
        RF(I,53) = RF(I,53)*C(I,5);
        RF(I,54) = RF(I,54)*C(I,5);
        RF(I,55) = RF(I,55)*C(I,7);
        RF(I,56) = RF(I,56)*C(I,11);
        RF(I,59) = RF(I,59)*C(I,22);
        RF(I,60) = RF(I,60)*C(I,2);
        RF(I,61) = RF(I,61)*C(I,3);
        RF(I,62) = RF(I,62)*C(I,3);
        RF(I,63) = RF(I,63)*C(I,5);
        RF(I,64) = RF(I,64)*C(I,1);
        RF(I,65) = RF(I,65)*C(I,4);
        RF(I,66) = RF(I,66)*C(I,4);
        RF(I,67) = RF(I,67)*C(I,6);
        RF(I,68) = RF(I,68)*C(I,11);
        RF(I,69) = RF(I,69)*C(I,12);
        RF(I,70) = RF(I,70)*C(I,12);
        RF(I,71) = RF(I,71)*C(I,13)*C(I,2);
        RF(I,72) = RF(I,72)*C(I,13)*C(I,2);
        RF(I,73) = RF(I,73)*C(I,13)*C(I,3);
        RF(I,74) = RF(I,74)*C(I,13)*C(I,5);
        RF(I,75) = RF(I,75)*C(I,13)*C(I,4);
        RF(I,76) = RF(I,76)*C(I,13)*C(I,7);
        RF(I,77) = RF(I,77)*C(I,13);
        RF(I,78) = RF(I,78)*C(I,9)*C(I,2);
        RF(I,79) = RF(I,79)*C(I,9)*C(I,3);
        RF(I,80) = RF(I,80)*C(I,9)*C(I,5);
        RF(I,81) = RF(I,81)*C(I,9)*C(I,5);
        RF(I,82) = RF(I,82)*C(I,9)*C(I,4);
        RF(I,83) = RF(I,83)*C(I,9)*C(I,4);
        RF(I,84) = RF(I,84)*C(I,9)*C(I,7);
        RF(I,85) = RF(I,85)*C(I,9)*C(I,7);
        RF(I,86) = RF(I,86)*C(I,9)*C(I,8);
        RF(I,87) = RF(I,87)*C(I,9);
        RF(I,88) = RF(I,88)*C(I,9);
        RF(I,89) = RF(I,89)*C(I,9);
        RF(I,90) = RF(I,90)*C(I,9)*C(I,13);
        RF(I,91) = RF(I,91)*C(I,9);
        RF(I,92) = RF(I,92)*C(I,9);
        RF(I,93) = RF(I,93)*C(I,9)*C(I,9);
        RF(I,94) = RF(I,94)*C(I,9)*C(I,9);
        RF(I,95) = RF(I,95)*C(I,9)*C(I,17);
        RF(I,96) = RF(I,96)*C(I,2);
        RF(I,97) = RF(I,97)*C(I,2);
        RF(I,98) = RF(I,98)*C(I,2);
        RF(I,99) = RF(I,99)*C(I,3);
        RF(I,100) = RF(I,100)*C(I,5);
        RF(I,101) = RF(I,101)*C(I,4);
        RF(I,102) = RF(I,102)*C(I,10)*C(I,2);
        RF(I,103) = RF(I,103)*C(I,10)*C(I,3);
        RF(I,104) = RF(I,104)*C(I,10)*C(I,5);
        RF(I,105) = RF(I,105)*C(I,10);
        RF(I,106) = RF(I,106)*C(I,10);
        RF(I,107) = RF(I,107)*C(I,10);
        RF(I,108) = RF(I,108)*C(I,17)*C(I,2);
        RF(I,109) = RF(I,109)*C(I,17)*C(I,3);
        RF(I,110) = RF(I,110)*C(I,17)*C(I,4);
        RF(I,111) = RF(I,111)*C(I,17);
        RF(I,112) = RF(I,112)*C(I,17);
        RF(I,113) = RF(I,113)*C(I,17)*C(I,17);
        RF(I,114) = RF(I,114)*C(I,14);
        RF(I,116) = RF(I,116)*C(I,14)*C(I,3);
        RF(I,117) = RF(I,117)*C(I,14)*C(I,3);
        RF(I,118) = RF(I,118)*C(I,14)*C(I,5);
        RF(I,119) = RF(I,119)*C(I,14)*C(I,5);
        RF(I,120) = RF(I,120)*C(I,14);
        RF(I,121) = RF(I,121)*CTB(I,121)*C(I,14)*C(I,9);
        RF(I,122) = RF(I,122)*C(I,2);
        RF(I,123) = RF(I,123)*C(I,3);
        RF(I,124) = RF(I,124)*C(I,5);
        RF(I,125) = RF(I,125)*C(I,4);
        RF(I,126) = RF(I,126)*C(I,18)*C(I,2);
        RF(I,127) = RF(I,127)*C(I,18)*C(I,2);
        RF(I,128) = RF(I,128)*C(I,18)*C(I,2);
        RF(I,129) = RF(I,129)*C(I,18)*C(I,3);
        RF(I,130) = RF(I,130)*C(I,18)*C(I,3);
        RF(I,131) = RF(I,131)*C(I,18)*C(I,5);
        RF(I,132) = RF(I,132)*C(I,2);
        RF(I,133) = RF(I,133)*C(I,2);
        RF(I,134) = RF(I,134)*C(I,2);
        RF(I,135) = RF(I,135)*C(I,3);
        RF(I,136) = RF(I,136)*C(I,3);
        RF(I,137) = RF(I,137)*C(I,5);
        RF(I,138) = RF(I,138)*C(I,4);
        RF(I,139) = RF(I,139)*C(I,4);
        RF(I,140) = RF(I,140)*C(I,4);
        RF(I,141) = RF(I,141)*C(I,7);
        RF(I,142) = RF(I,142)*C(I,8);
        RF(I,144) = RF(I,144)*C(I,9);
        RF(I,145) = RF(I,145)*C(I,9);
        RF(I,146) = RF(I,146)*C(I,9);
        RF(I,148) = RF(I,148)*C(I,2);
        RF(I,149) = RF(I,149)*C(I,2);
        RF(I,150) = RF(I,150)*C(I,2);
        RF(I,151) = RF(I,151)*C(I,3);
        RF(I,152) = RF(I,152)*C(I,5);
        RF(I,153) = RF(I,153)*C(I,4);
        RF(I,154) = RF(I,154)*C(I,4);
        RF(I,155) = RF(I,155)*C(I,15);
        RF(I,156) = RF(I,156)*C(I,15)*C(I,2);
        RF(I,157) = RF(I,157)*C(I,15)*C(I,2);
        RF(I,158) = RF(I,158)*C(I,15)*C(I,3);
        RF(I,159) = RF(I,159)*C(I,15)*C(I,3);
        RF(I,160) = RF(I,160)*C(I,15)*C(I,3);
        RF(I,161) = RF(I,161)*C(I,15)*C(I,5);
        RF(I,162) = RF(I,162)*C(I,15)*C(I,4);
        RF(I,163) = RF(I,163)*C(I,15)*C(I,7);
        RF(I,164) = RF(I,164)*C(I,15);
        RF(I,165) = RF(I,165)*C(I,15);
        RF(I,166) = RF(I,166)*C(I,15);
        RF(I,167) = RF(I,167)*C(I,15);
        RF(I,168) = RF(I,168)*C(I,15)*C(I,9);
        RF(I,169) = RF(I,169)*C(I,15)*C(I,9);
        RF(I,170) = RF(I,170)*C(I,2);
        RF(I,171) = RF(I,171)*C(I,2);
        RF(I,172) = RF(I,172)*C(I,3);
        RF(I,173) = RF(I,173)*C(I,3);
        RF(I,174) = RF(I,174)*C(I,4);
        RF(I,175) = RF(I,175)*C(I,7);
        RF(I,176) = RF(I,176)*C(I,7);
        RF(I,177) = RF(I,177)*C(I,7);
        RF(I,178) = RF(I,178)*C(I,8);
        RF(I,180) = RF(I,180)*C(I,16)*C(I,2);
        RF(I,181) = RF(I,181)*C(I,16)*C(I,3);
        RF(I,182) = RF(I,182)*C(I,16)*C(I,5);
        RF(I,183) = RF(I,183)*C(I,16);
        RF(I,184) = RF(I,184)*C(I,16)*C(I,9);
        RF(I,185) = RF(I,185)*C(I,20)*C(I,2);
        RF(I,186) = RF(I,186)*C(I,20)*C(I,2);
        RF(I,187) = RF(I,187)*C(I,20)*C(I,7);
        RF(I,188) = RF(I,188)*C(I,20)*C(I,7);
        RF(I,189) = RF(I,189)*C(I,20);
        RF(I,190) = RF(I,190)*C(I,21)*C(I,2);
        RF(I,191) = RF(I,191)*C(I,21)*C(I,2);
        RF(I,192) = RF(I,192)*C(I,21)*C(I,2);
        RF(I,193) = RF(I,193)*C(I,21)*C(I,3);
        RF(I,194) = RF(I,194)*C(I,21)*C(I,3);
        RF(I,195) = RF(I,195)*C(I,21)*C(I,3);
        RF(I,196) = RF(I,196)*C(I,21)*C(I,5);
        RF(I,197) = RF(I,197)*C(I,21)*C(I,7);
        RF(I,198) = RF(I,198)*C(I,21)*C(I,9);
        RF(I,199) = RF(I,199)*C(I,2);
        RF(I,200) = RF(I,200)*C(I,2);
        RF(I,201) = RF(I,201)*C(I,3);
        RF(I,202) = RF(I,202)*C(I,5);
        RF(I,203) = RF(I,203)*C(I,4);
        RF(I,204) = RF(I,204)*C(I,7);
        RF(I,205) = RF(I,205)*C(I,9);
    }

    #pragma vector aligned
    for (I=1; I<=VL; I++) 
    {
        RB(I,1) = RB(I,1)*C(I,3)*C(I,5);
        RB(I,2) = RB(I,2)*C(I,2)*C(I,5);
        RB(I,3) = RB(I,3)*C(I,2)*C(I,6);
        RB(I,4) = RB(I,4)*C(I,3)*C(I,6);
        RB(I,5) = RB(I,5)*CTB(I,5)*C(I,1);
        RB(I,6) = RB(I,6)*C(I,1)*C(I,1);
        RB(I,7) = RB(I,7)*C(I,1)*C(I,6);
        RB(I,8) = RB(I,8)*C(I,1)*C(I,12);
        RB(I,9) = RB(I,9)*CTB(I,9)*C(I,6);
        RB(I,10) = RB(I,10)*CTB(I,10)*C(I,5);
        RB(I,11) = RB(I,11)*CTB(I,11)*C(I,4);
        RB(I,12) = RB(I,12)*CTB(I,12)*C(I,7);
        RB(I,13) = RB(I,13)*C(I,7)*C(I,4);
        RB(I,14) = RB(I,14)*C(I,7)*C(I,6);
        RB(I,15) = RB(I,15)*C(I,7)*C(I,22);
        RB(I,16) = RB(I,16)*C(I,8);
        RB(I,17) = RB(I,17)*C(I,3)*C(I,6);
        RB(I,18) = RB(I,18)*C(I,4)*C(I,1);
        RB(I,19) = RB(I,19)*C(I,5)*C(I,5);
        RB(I,20) = RB(I,20)*C(I,5)*C(I,4);
        RB(I,21) = RB(I,21)*C(I,4)*C(I,6);
        RB(I,22) = RB(I,22)*C(I,4)*C(I,8);
        RB(I,23) = RB(I,23)*C(I,4)*C(I,8);
        RB(I,24) = RB(I,24)*C(I,7)*C(I,1);
        RB(I,25) = RB(I,25)*C(I,5)*C(I,6);
        RB(I,26) = RB(I,26)*C(I,5)*C(I,7);
        RB(I,27) = RB(I,27)*C(I,7)*C(I,6);
        RB(I,28) = RB(I,28)*C(I,7)*C(I,6);
        RB(I,29) = RB(I,29)*CTB(I,29)*C(I,12);
        RB(I,30) = RB(I,30)*C(I,12)*C(I,2);
        RB(I,31) = RB(I,31)*C(I,13);
        RB(I,32) = RB(I,32)*C(I,12)*C(I,3);
        RB(I,33) = RB(I,33)*C(I,12)*C(I,5);
        RB(I,34) = RB(I,34)*C(I,11)*C(I,2);
        RB(I,35) = RB(I,35)*C(I,2);
        RB(I,36) = RB(I,36)*C(I,2);
        RB(I,37) = RB(I,37)*C(I,13)*C(I,2);
        RB(I,38) = RB(I,38)*C(I,3);
        RB(I,39) = RB(I,39)*C(I,17);
        RB(I,40) = RB(I,40)*C(I,11);
        RB(I,41) = RB(I,41)*C(I,13);
        RB(I,42) = RB(I,42)*C(I,11)*C(I,1);
        RB(I,43) = RB(I,43)*C(I,11)*C(I,5);
        RB(I,44) = RB(I,44)*C(I,12)*C(I,2);
        RB(I,45) = RB(I,45)*C(I,11)*C(I,6);
        RB(I,46) = RB(I,46)*CTB(I,46)*C(I,11)*C(I,2);
        RB(I,47) = RB(I,47)*C(I,11)*C(I,7);
        RB(I,48) = RB(I,48)*C(I,9);
        RB(I,49) = RB(I,49)*C(I,2)*C(I,9);
        RB(I,50) = RB(I,50)*C(I,2);
        RB(I,51) = RB(I,51)*C(I,5);
        RB(I,52) = RB(I,52)*C(I,12)*C(I,2)*C(I,2);
        RB(I,53) = RB(I,53)*C(I,13)*C(I,2);
        RB(I,54) = RB(I,54)*C(I,6);
        RB(I,55) = RB(I,55)*C(I,13)*C(I,5);
        RB(I,56) = RB(I,56)*C(I,18);
        RB(I,57) = RB(I,57)*C(I,14)*C(I,2);
        RB(I,58) = RB(I,58)*C(I,14)*C(I,1);
        RB(I,59) = RB(I,59)*C(I,22);
        RB(I,60) = RB(I,60)*C(I,1);
        RB(I,61) = RB(I,61)*C(I,11)*C(I,1);
        RB(I,62) = RB(I,62)*C(I,2);
        RB(I,63) = RB(I,63)*C(I,13)*C(I,2);
        RB(I,64) = RB(I,64)*C(I,9)*C(I,2);
        RB(I,65) = RB(I,65)*C(I,2)*C(I,5)*C(I,11);
        RB(I,66) = RB(I,66)*C(I,11)*C(I,6);
        RB(I,67) = RB(I,67)*C(I,6);
        RB(I,68) = RB(I,68)*C(I,11);
        RB(I,69) = RB(I,69)*C(I,12);
        RB(I,70) = RB(I,70)*C(I,13)*C(I,11);
        RB(I,72) = RB(I,72)*C(I,1);
        RB(I,73) = RB(I,73)*C(I,5);
        RB(I,74) = RB(I,74)*C(I,6);
        RB(I,75) = RB(I,75)*C(I,7);
        RB(I,76) = RB(I,76)*C(I,8);
        RB(I,77) = RB(I,77)*C(I,18)*C(I,2);
        RB(I,78) = RB(I,78)*C(I,10);
        RB(I,79) = RB(I,79)*C(I,13)*C(I,2);
        RB(I,80) = RB(I,80)*C(I,6);
        RB(I,81) = RB(I,81)*C(I,6);
        RB(I,82) = RB(I,82)*C(I,3);
        RB(I,83) = RB(I,83)*C(I,5)*C(I,13);
        RB(I,84) = RB(I,84)*C(I,10)*C(I,4);
        RB(I,85) = RB(I,85)*C(I,5);
        RB(I,86) = RB(I,86)*C(I,10)*C(I,7);
        RB(I,87) = RB(I,87)*C(I,2);
        RB(I,88) = RB(I,88)*C(I,10)*C(I,11);
        RB(I,89) = RB(I,89)*C(I,19);
        RB(I,90) = RB(I,90)*C(I,10);
        RB(I,91) = RB(I,91)*C(I,15)*C(I,2);
        RB(I,92) = RB(I,92)*C(I,15)*C(I,2);
        RB(I,93) = RB(I,93)*C(I,16);
        RB(I,94) = RB(I,94)*C(I,2);
        RB(I,95) = RB(I,95)*C(I,15)*C(I,11);
        RB(I,96) = RB(I,96)*C(I,13)*C(I,1);
        RB(I,97) = RB(I,97)*C(I,9)*C(I,5);
        RB(I,98) = RB(I,98)*C(I,6);
        RB(I,99) = RB(I,99)*C(I,13)*C(I,5);
        RB(I,100) = RB(I,100)*C(I,13)*C(I,6);
        RB(I,101) = RB(I,101)*C(I,13)*C(I,7);
        RB(I,102) = RB(I,102)*C(I,9)*C(I,1);
        RB(I,103) = RB(I,103)*C(I,9)*C(I,5);
        RB(I,104) = RB(I,104)*C(I,9)*C(I,6);
        RB(I,105) = RB(I,105)*C(I,15)*C(I,2);
        RB(I,106) = RB(I,106)*C(I,9)*C(I,9);
        RB(I,107) = RB(I,107)*C(I,9)*C(I,9);
        RB(I,108) = RB(I,108)*C(I,11);
        RB(I,109) = RB(I,109)*C(I,2)*C(I,11)*C(I,11);
        RB(I,110) = RB(I,110)*C(I,5)*C(I,11)*C(I,11);
        RB(I,111) = RB(I,111)*C(I,14)*C(I,11);
        RB(I,112) = RB(I,112)*C(I,11);
        RB(I,113) = RB(I,113)*C(I,14)*C(I,11)*C(I,11);
        RB(I,115) = RB(I,115)*C(I,14)*C(I,2);
        RB(I,116) = RB(I,116)*C(I,17)*C(I,2);
        RB(I,117) = RB(I,117)*C(I,11);
        RB(I,118) = RB(I,118)*C(I,18)*C(I,2);
        RB(I,119) = RB(I,119)*C(I,9)*C(I,11);
        RB(I,120) = RB(I,120)*C(I,11);
        RB(I,121) = RB(I,121)*CTB(I,121)*C(I,20);
        RB(I,122) = RB(I,122)*C(I,14)*C(I,2);
        RB(I,123) = RB(I,123)*C(I,11);
        RB(I,124) = RB(I,124)*C(I,18)*C(I,2);
        RB(I,125) = RB(I,125)*C(I,12);
        RB(I,127) = RB(I,127)*C(I,17)*C(I,1);
        RB(I,128) = RB(I,128)*C(I,9)*C(I,11);
        RB(I,129) = RB(I,129)*C(I,17)*C(I,5);
        RB(I,130) = RB(I,130)*C(I,12);
        RB(I,131) = RB(I,131)*C(I,17)*C(I,6);
        RB(I,132) = RB(I,132)*C(I,15);
        RB(I,133) = RB(I,133)*C(I,14)*C(I,1);
        RB(I,134) = RB(I,134)*C(I,1);
        RB(I,135) = RB(I,135)*C(I,18)*C(I,2);
        RB(I,136) = RB(I,136)*C(I,9)*C(I,11);
        RB(I,137) = RB(I,137)*C(I,14)*C(I,6);
        RB(I,138) = RB(I,138)*C(I,14)*C(I,7);
        RB(I,139) = RB(I,139)*C(I,3);
        RB(I,140) = RB(I,140)*C(I,13);
        RB(I,141) = RB(I,141)*C(I,5);
        RB(I,142) = RB(I,142)*C(I,15)*C(I,7);
        RB(I,143) = RB(I,143)*C(I,15)*C(I,11);
        RB(I,144) = RB(I,144)*C(I,14)*C(I,10);
        RB(I,145) = RB(I,145)*C(I,21);
        RB(I,146) = RB(I,146)*C(I,20)*C(I,2);
        RB(I,147) = RB(I,147)*C(I,9)*C(I,11);
        RB(I,148) = RB(I,148)*C(I,19);
        RB(I,149) = RB(I,149)*C(I,9);
        RB(I,150) = RB(I,150)*C(I,18)*C(I,1);
        RB(I,151) = RB(I,151)*C(I,18)*C(I,5);
        RB(I,152) = RB(I,152)*C(I,18)*C(I,6);
        RB(I,153) = RB(I,153)*C(I,18)*C(I,7);
        RB(I,154) = RB(I,154)*C(I,13)*C(I,11)*C(I,5);
        RB(I,155) = RB(I,155)*C(I,1);
        RB(I,157) = RB(I,157)*C(I,1);
        RB(I,158) = RB(I,158)*C(I,5);
        RB(I,159) = RB(I,159)*C(I,9);
        RB(I,160) = RB(I,160)*C(I,13);
        RB(I,161) = RB(I,161)*C(I,6);
        RB(I,162) = RB(I,162)*C(I,7);
        RB(I,163) = RB(I,163)*C(I,19)*C(I,5);
        RB(I,164) = RB(I,164)*C(I,11);
        RB(I,165) = RB(I,165)*C(I,20)*C(I,2);
        RB(I,166) = RB(I,166)*C(I,10);
        RB(I,167) = RB(I,167)*C(I,20)*C(I,2);
        RB(I,168) = RB(I,168)*C(I,10);
        RB(I,170) = RB(I,170)*C(I,16);
        RB(I,171) = RB(I,171)*C(I,15)*C(I,1);
        RB(I,172) = RB(I,172)*C(I,9)*C(I,13);
        RB(I,173) = RB(I,173)*C(I,19)*C(I,2);
        RB(I,174) = RB(I,174)*C(I,15)*C(I,7);
        RB(I,175) = RB(I,175)*C(I,16)*C(I,4);
        RB(I,176) = RB(I,176)*C(I,15)*C(I,8);
        RB(I,177) = RB(I,177)*C(I,9)*C(I,13)*C(I,5);
        RB(I,178) = RB(I,178)*C(I,16)*C(I,7);
        RB(I,179) = RB(I,179)*C(I,16)*C(I,11);
        RB(I,180) = RB(I,180)*C(I,1);
        RB(I,181) = RB(I,181)*C(I,5);
        RB(I,182) = RB(I,182)*C(I,6);
        RB(I,183) = RB(I,183)*C(I,9);
        RB(I,184) = RB(I,184)*C(I,10);
        RB(I,185) = RB(I,185)*C(I,21);
        RB(I,186) = RB(I,186)*C(I,10);
        RB(I,187) = RB(I,187)*C(I,21)*C(I,4);
        RB(I,188) = RB(I,188)*C(I,5)*C(I,13);
        RB(I,189) = RB(I,189)*C(I,21)*C(I,11);
        RB(I,191) = RB(I,191)*C(I,15)*C(I,9);
        RB(I,192) = RB(I,192)*C(I,20)*C(I,1);
        RB(I,193) = RB(I,193)*C(I,18)*C(I,9)*C(I,2);
        RB(I,195) = RB(I,195)*C(I,20)*C(I,5);
        RB(I,196) = RB(I,196)*C(I,20)*C(I,6);
        RB(I,197) = RB(I,197)*C(I,20)*C(I,8);
        RB(I,198) = RB(I,198)*C(I,20)*C(I,10);
        RB(I,199) = RB(I,199)*C(I,9);
        RB(I,200) = RB(I,200)*C(I,21)*C(I,1);
        RB(I,201) = RB(I,201)*C(I,13);
        RB(I,202) = RB(I,202)*C(I,21)*C(I,6);
        RB(I,203) = RB(I,203)*C(I,21)*C(I,7);
        RB(I,204) = RB(I,204)*C(I,5)*C(I,13);
        RB(I,205) = RB(I,205)*C(I,10)*C(I,21);
        RB(I,206) = RB(I,206)*C(I,20)*C(I,9);
    }
}


template <class real, int MAXVL>
__declspec(target(mic))  void 
ratx_i_(int *VLp, real * RESTRICT T, real * RESTRICT C, real * RESTRICT RF,
        real * RESTRICT RB, real * RESTRICT RKLOW)
{

    ALIGN64        real TI[MAXVL], CTB[MAXVL*206], CTOT[MAXVL], PR[MAXVL],
                   PCOR[MAXVL], PRLOG[MAXVL], FCENT0[MAXVL], FCENT1[MAXVL],
                   FCENT2[MAXVL], FCLOG[MAXVL], XN[MAXVL], CPRLOG[MAXVL],
                   FLOG[MAXVL], FC[MAXVL], FCENT[MAXVL], ALOGT[MAXVL], 
                   vecof10[MAXVL];

    int VL = *VLp;
    int I, K;
    const real SMALL= floatMin<real>();//FLT_MIN; // 1.e-200;

    for (I=1; I<=VL; I++) vecof10(I) =10.;
    for (I=1; I<=VL; I++) TI(I)=1./T(I);

    vrda_log_(VLp, T, ALOGT);
    //
    //    third-body reactions
    //
    for (I=1; I<=VL; I++) CTOT(I) = 0.0;
    for (K = 1; K<=22; K++) {
        for (I=1; I<=VL; I++) {
            CTOT(I) = CTOT(I) + C(I,K);
        }
    }

    for (I=1; I<=VL; I++) CTB(I,5)  = CTOT(I) - C(I,1) - C(I,6) + C(I,10) - C(I,12) 
        + 2.e0*C(I,16) + 2.e0*C(I,14) + 2.e0*C(I,15);
    for (I=1; I<=VL; I++) CTB(I,9)  = CTOT(I) - 2.7e-1*C(I,1) + 2.65e0*C(I,6) + C(I,10) 
        + 2.e0*C(I,16) + 2.e0*C(I,14) + 2.e0*C(I,15);
    for (I=1; I<=VL; I++) CTB(I,10) = CTOT(I) + C(I,1) + 5.e0*C(I,6) + C(I,10)
        + 5.e-1*C(I,11) + C(I,12) 
            + 2.e0*C(I,16) + 2.e0*C(I,14) + 2.e0*C(I,15);
    for (I=1; I<=VL; I++) CTB(I,31) = CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,39) = CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,41) = CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,46) = CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,48) = CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,56) = CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,71) = CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,78) = CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,89) = CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,93) = CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,115)= CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,126)= CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,132)= CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,145)= CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,148)= CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,155)= CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,156)= CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,170)= CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,185)= CTB(I,10);
    for (I=1; I<=VL; I++) CTB(I,114)= CTOT(I)+C(I,1)+5.e0*C(I,6)+C(I,10) 
        +5.e-1*C(I,11)+C(I,12)  
            +2.e0*C(I,16)+1.5e0*C(I,14)+1.5e0*C(I,15);
    for (I=1; I<=VL; I++) CTB(I,11) = CTOT(I)+1.4e0*C(I,1)+1.44e1*C(I,6)+C(I,10) 
        +7.5e-1*C(I,11)  
            +2.6e0*C(I,12)+2.e0*C(I,16)+2.e0*C(I,14)   
            +2.e0*C(I,15);
    for (I=1; I<=VL; I++) CTB(I,12) = CTOT(I) - C(I,4) - C(I,6) - 2.5e-1*C(I,11) 
        +5.e-1*C(I,12)  
            +5.e-1*C(I,16) - C(I,22)+2.e0*C(I,14) 
            +2.e0*C(I,15);
    for (I=1; I<=VL; I++) CTB(I,16) = CTOT(I)+C(I,1)+5.e0*C(I,6)+C(I,10)  
        +5.e-1*C(I,11) +C(I,12)  
            +2.e0*C(I,16)+2.e0*C(I,14)+2.e0*C(I,15);
    for (I=1; I<=VL; I++) CTB(I,29) = CTOT(I)+C(I,1)+5.e0*C(I,4)+5.e0*C(I,6)+C(I,10)  
        +5.e-1*C(I,11)+2.5e0*C(I,12)+2.e0*C(I,16)  
            +2.e0*C(I,14)+2.e0*C(I,15);
    for (I=1; I<=VL; I++) CTB(I,121)= CTOT(I); 
    for (I=1; I<=VL; I++) CTB(I,190)= CTOT(I)+C(I,1)+5.e0*C(I,6)+C(I,10)
        +5.e-1*C(I,11)+C(I,12)+2.e0*C(I,16);

    //     If fall-off (pressure correction):
#define ALOG log

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,1) * CTB(I,16) / RF(I,16);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log(MAX(PR(I),SMALL));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/9.4e1);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.756e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.182e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 2.654e-1*FCENT0(I) +7.346e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,16) = RF(I,16) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,16) = RB(I,16) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,2) * CTB(I,31) / RF(I,31);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( MAX(PR(I),SMALL));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.97e2);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.54e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-1.03e4*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 6.8e-2*FCENT0(I) +9.32e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,31) = RF(I,31) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,31) = RB(I,31) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,3) * CTB(I,39) / RF(I,39);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/2.37e2);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.652e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.069e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 4.243e-1*FCENT0(I) +5.757e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( MAX(FCENT(I),SMALL));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,39) = RF(I,39) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,39) = RB(I,39) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,4) * CTB(I,41) / RF(I,41);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/2.71e2);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/2.755e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-6.57e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 2.176e-1*FCENT0(I) +7.824e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,41) = RF(I,41) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,41) = RB(I,41) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,5) * CTB(I,48) / RF(I,48);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/7.8e1);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.995e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.59e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 3.2e-1*FCENT0(I) +6.8e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log((MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,48) = RF(I,48) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,48) = RB(I,48) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,6) * CTB(I,56) / RF(I,56);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/2.75e2);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.226e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.185e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 4.093e-1*FCENT0(I) +5.907e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    //       vrda_exp_(VLp,FC,FC);;
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,56) = RF(I,56) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,56) = RB(I,56) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,7) * CTB(I,71) / RF(I,71);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/9.4e1);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.555e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-4.2e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 2.42e-1*FCENT0(I) +7.58e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( MAX(FCENT(I),SMALL));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,71) = RF(I,71) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,71) = RB(I,71) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,8) * CTB(I,78) / RF(I,78);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/7.4e1);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/2.941e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-6.964e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 2.17e-1*FCENT0(I) +7.83e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,78) = RF(I,78) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,78) = RB(I,78) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,9) * CTB(I,89) / RF(I,89);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.3076e1);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/2.078e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.093e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 3.827e-1*FCENT0(I) +6.173e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,89) = RF(I,89) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,89) = RB(I,89) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,10) * CTB(I,93) / RF(I,93);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.51e2);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.038e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-4.97e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 4.675e-1*FCENT0(I) +5.325e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,93) = RF(I,93) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,93) = RB(I,93) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,11) * CTB(I,114) / RF(I,114);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) RF(I,114) = RF(I,114) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,114) = RB(I,114) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,12) * CTB(I,115) / RF(I,115);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/5.3837e3);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/4.2932e0);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(7.95e-2*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = -9.816e-1*FCENT0(I) +1.9816e0*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,115) = RF(I,115) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,115) = RB(I,115) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,13) * CTB(I,126) / RF(I,126);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT(I) = 6.63e-1*EXP(-T(I)/1.707e3) + 3.37e-1*EXP(-T(I)/3.2e3)
        + EXP(-4.131e3*TI(I));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.707e3);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/3.2e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-4.131e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 6.63e-1*FCENT0(I) +3.37e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,126) = RF(I,126) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,126) = RB(I,126) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,14) * CTB(I,132) / RF(I,132);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/2.075e2);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/2.663e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-6.095e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 2.18e-1*FCENT0(I) +7.82e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,132) = RF(I,132) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,132) = RB(I,132) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,15) * CTB(I,145) / RF(I,145);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.3406e3);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/6.e4);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-1.01398e4*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 8.25e-1*FCENT0(I) + 1.75e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,145) = RF(I,145) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,145) = RB(I,145) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,16) * CTB(I,148) / RF(I,148);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/8.9e3);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/4.35e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-7.244e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 4.5e-1*FCENT0(I) + 5.5e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,148) = RF(I,148) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,148) = RB(I,148) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,17) * CTB(I,155) / RF(I,155);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.8e2);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.035e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-5.417e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 2.655e-1*FCENT0(I) + 7.345e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,155) = RF(I,155) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,155) = RB(I,155) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,18) * CTB(I,156) / RF(I,156);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( MAX(PR(I),SMALL));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/2.1e2);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/9.84e2);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-4.374e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 2.47e-2*FCENT0(I) + 9.753e-1*FCENT1(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,156) = RF(I,156) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,156) = RB(I,156) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,19) * CTB(I,170) / RF(I,170);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( (MAX(PR(I),SMALL)));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.25e2);
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/2.219e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-6.882e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 1.578e-1*FCENT0(I) + 8.422e-1*FCENT1(I) + FCENT2(I);

    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,170) = RF(I,170) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,170) = RB(I,170) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,20) * CTB(I,185) / RF(I,185);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( MAX(PR(I),SMALL));
    for (I=1; I<=VL; I++) FCENT0(I) = EXP(-T(I)/1.0966e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-6.8595e3*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = 9.8e-1*FCENT0(I) + 2.e-2*FCENT0(I) + FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( (MAX(FCENT(I),SMALL)));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,185) = RF(I,185) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,185) = RB(I,185) * PCOR(I);

    for (I=1; I<=VL; I++) PR(I) = RKLOW(I,21) * CTB(I,190) / RF(I,190);
    for (I=1; I<=VL; I++) PCOR(I) = PR(I) / (1.0 + PR(I));
    for (I=1; I<=VL; I++) PRLOG(I) = RECIP*log( MAX(PR(I),SMALL));
    for (I=1; I<=VL; I++) FCENT1(I) = EXP(-T(I)/1.31e3);
    for (I=1; I<=VL; I++) FCENT2(I) = EXP(-4.8097e4*TI(I));
    for (I=1; I<=VL; I++) FCENT(I) = FCENT1(I)+FCENT2(I);
    for (I=1; I<=VL; I++) FCLOG(I) = RECIP*log( MAX(FCENT(I),SMALL));
    for (I=1; I<=VL; I++) XN(I)    = 0.75 - 1.27*FCLOG(I);
    for (I=1; I<=VL; I++) CPRLOG(I)= PRLOG(I) - (0.4 + 0.67*FCLOG(I));
    for (I=1; I<=VL; I++) FLOG(I) = FCLOG(I)/(1.0 + (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I)))*
            (CPRLOG(I)/(XN(I)-0.14*CPRLOG(I))));
    for (I=1; I<=VL; I++) FC(I) = EXP(ALOG(10.0)*FLOG(I));
    for (I=1; I<=VL; I++) PCOR(I) = FC(I) * PCOR(I);
    for (I=1; I<=VL; I++) RF(I,190) = RF(I,190) * PCOR(I);
    for (I=1; I<=VL; I++) RB(I,190) = RB(I,190) * PCOR(I);

    for (I=1; I<=VL; I++) 
    {
        RF(I,1) = RF(I,1)*C(I,2)*C(I,4);
        RF(I,2) = RF(I,2)*C(I,3)*C(I,1);
        RF(I,3) = RF(I,3)*C(I,5)*C(I,1);
        RF(I,4) = RF(I,4)*C(I,5)*C(I,5);
        RF(I,5) = RF(I,5)*CTB(I,5)*C(I,2)*C(I,2);
        RF(I,6) = RF(I,6)*C(I,2)*C(I,2)*C(I,1);
        RF(I,7) = RF(I,7)*C(I,2)*C(I,2)*C(I,6);
        RF(I,8) = RF(I,8)*C(I,2)*C(I,2)*C(I,12);
        RF(I,9) = RF(I,9)*CTB(I,9)*C(I,2)*C(I,5);
        RF(I,10) = RF(I,10)*CTB(I,10)*C(I,3)*C(I,2);
        RF(I,11) = RF(I,11)*CTB(I,11)*C(I,3)*C(I,3);
        RF(I,12) = RF(I,12)*CTB(I,12)*C(I,2)*C(I,4);
        RF(I,13) = RF(I,13)*C(I,2)*C(I,4)*C(I,4);
        RF(I,14) = RF(I,14)*C(I,2)*C(I,4)*C(I,6);
        RF(I,15) = RF(I,15)*C(I,2)*C(I,4)*C(I,22);
        RF(I,16) = RF(I,16)*C(I,5)*C(I,5);
        RF(I,17) = RF(I,17)*C(I,7)*C(I,2);
        RF(I,18) = RF(I,18)*C(I,7)*C(I,2);
        RF(I,19) = RF(I,19)*C(I,7)*C(I,2);
        RF(I,20) = RF(I,20)*C(I,7)*C(I,3);
        RF(I,21) = RF(I,21)*C(I,7)*C(I,5);
        RF(I,22) = RF(I,22)*C(I,7)*C(I,7);
        RF(I,23) = RF(I,23)*C(I,7)*C(I,7);
        RF(I,24) = RF(I,24)*C(I,8)*C(I,2);
        RF(I,25) = RF(I,25)*C(I,8)*C(I,2);
        RF(I,26) = RF(I,26)*C(I,8)*C(I,3);
        RF(I,27) = RF(I,27)*C(I,8)*C(I,5);
        RF(I,28) = RF(I,28)*C(I,8)*C(I,5);
        RF(I,29) = RF(I,29)*CTB(I,29)*C(I,11)*C(I,3);
        RF(I,30) = RF(I,30)*C(I,11)*C(I,5);
        RF(I,31) = RF(I,31)*C(I,11)*C(I,1);
        RF(I,32) = RF(I,32)*C(I,11)*C(I,4);
        RF(I,33) = RF(I,33)*C(I,11)*C(I,7);
        RF(I,34) = RF(I,34)*C(I,3);
        RF(I,35) = RF(I,35)*C(I,5);
        RF(I,36) = RF(I,36)*C(I,1);
        RF(I,37) = RF(I,37)*C(I,6);
        RF(I,38) = RF(I,38)*C(I,4);
        RF(I,39) = RF(I,39)*C(I,11);
        RF(I,40) = RF(I,40)*C(I,12);
        RF(I,41) = RF(I,41)*C(I,2);
        RF(I,42) = RF(I,42)*C(I,2);
        RF(I,43) = RF(I,43)*C(I,3);
        RF(I,44) = RF(I,44)*C(I,3);
        RF(I,45) = RF(I,45)*C(I,5);
        RF(I,46) = RF(I,46)*CTB(I,46);
        RF(I,47) = RF(I,47)*C(I,4);
        RF(I,48) = RF(I,48)*C(I,2);
        RF(I,49) = RF(I,49)*C(I,1);
        RF(I,50) = RF(I,50)*C(I,3);
        RF(I,51) = RF(I,51)*C(I,4);
        RF(I,52) = RF(I,52)*C(I,4);
        RF(I,53) = RF(I,53)*C(I,5);
        RF(I,54) = RF(I,54)*C(I,5);
        RF(I,55) = RF(I,55)*C(I,7);
        RF(I,56) = RF(I,56)*C(I,11);
        RF(I,59) = RF(I,59)*C(I,22);
        RF(I,60) = RF(I,60)*C(I,2);
        RF(I,61) = RF(I,61)*C(I,3);
        RF(I,62) = RF(I,62)*C(I,3);
        RF(I,63) = RF(I,63)*C(I,5);
        RF(I,64) = RF(I,64)*C(I,1);
        RF(I,65) = RF(I,65)*C(I,4);
        RF(I,66) = RF(I,66)*C(I,4);
        RF(I,67) = RF(I,67)*C(I,6);
        RF(I,68) = RF(I,68)*C(I,11);
        RF(I,69) = RF(I,69)*C(I,12);
        RF(I,70) = RF(I,70)*C(I,12);
        RF(I,71) = RF(I,71)*C(I,13)*C(I,2);
        RF(I,72) = RF(I,72)*C(I,13)*C(I,2);
        RF(I,73) = RF(I,73)*C(I,13)*C(I,3);
        RF(I,74) = RF(I,74)*C(I,13)*C(I,5);
        RF(I,75) = RF(I,75)*C(I,13)*C(I,4);
        RF(I,76) = RF(I,76)*C(I,13)*C(I,7);
        RF(I,77) = RF(I,77)*C(I,13);
        RF(I,78) = RF(I,78)*C(I,9)*C(I,2);
        RF(I,79) = RF(I,79)*C(I,9)*C(I,3);
        RF(I,80) = RF(I,80)*C(I,9)*C(I,5);
        RF(I,81) = RF(I,81)*C(I,9)*C(I,5);
        RF(I,82) = RF(I,82)*C(I,9)*C(I,4);
        RF(I,83) = RF(I,83)*C(I,9)*C(I,4);
        RF(I,84) = RF(I,84)*C(I,9)*C(I,7);
        RF(I,85) = RF(I,85)*C(I,9)*C(I,7);
        RF(I,86) = RF(I,86)*C(I,9)*C(I,8);
        RF(I,87) = RF(I,87)*C(I,9);
        RF(I,88) = RF(I,88)*C(I,9);
        RF(I,89) = RF(I,89)*C(I,9);
        RF(I,90) = RF(I,90)*C(I,9)*C(I,13);
        RF(I,91) = RF(I,91)*C(I,9);
        RF(I,92) = RF(I,92)*C(I,9);
        RF(I,93) = RF(I,93)*C(I,9)*C(I,9);
        RF(I,94) = RF(I,94)*C(I,9)*C(I,9);
        RF(I,95) = RF(I,95)*C(I,9)*C(I,17);
        RF(I,96) = RF(I,96)*C(I,2);
        RF(I,97) = RF(I,97)*C(I,2);
        RF(I,98) = RF(I,98)*C(I,2);
        RF(I,99) = RF(I,99)*C(I,3);
        RF(I,100) = RF(I,100)*C(I,5);
        RF(I,101) = RF(I,101)*C(I,4);
        RF(I,102) = RF(I,102)*C(I,10)*C(I,2);
        RF(I,103) = RF(I,103)*C(I,10)*C(I,3);
        RF(I,104) = RF(I,104)*C(I,10)*C(I,5);
        RF(I,105) = RF(I,105)*C(I,10);
        RF(I,106) = RF(I,106)*C(I,10);
        RF(I,107) = RF(I,107)*C(I,10);
        RF(I,108) = RF(I,108)*C(I,17)*C(I,2);
        RF(I,109) = RF(I,109)*C(I,17)*C(I,3);
        RF(I,110) = RF(I,110)*C(I,17)*C(I,4);
        RF(I,111) = RF(I,111)*C(I,17);
        RF(I,112) = RF(I,112)*C(I,17);
        RF(I,113) = RF(I,113)*C(I,17)*C(I,17);
        RF(I,114) = RF(I,114)*C(I,14);
        RF(I,116) = RF(I,116)*C(I,14)*C(I,3);
        RF(I,117) = RF(I,117)*C(I,14)*C(I,3);
        RF(I,118) = RF(I,118)*C(I,14)*C(I,5);
        RF(I,119) = RF(I,119)*C(I,14)*C(I,5);
        RF(I,120) = RF(I,120)*C(I,14);
        RF(I,121) = RF(I,121)*CTB(I,121)*C(I,14)*C(I,9);
        RF(I,122) = RF(I,122)*C(I,2);
        RF(I,123) = RF(I,123)*C(I,3);
        RF(I,124) = RF(I,124)*C(I,5);
        RF(I,125) = RF(I,125)*C(I,4);
        RF(I,126) = RF(I,126)*C(I,18)*C(I,2);
        RF(I,127) = RF(I,127)*C(I,18)*C(I,2);
        RF(I,128) = RF(I,128)*C(I,18)*C(I,2);
        RF(I,129) = RF(I,129)*C(I,18)*C(I,3);
        RF(I,130) = RF(I,130)*C(I,18)*C(I,3);
        RF(I,131) = RF(I,131)*C(I,18)*C(I,5);
        RF(I,132) = RF(I,132)*C(I,2);
        RF(I,133) = RF(I,133)*C(I,2);
        RF(I,134) = RF(I,134)*C(I,2);
        RF(I,135) = RF(I,135)*C(I,3);
        RF(I,136) = RF(I,136)*C(I,3);
        RF(I,137) = RF(I,137)*C(I,5);
        RF(I,138) = RF(I,138)*C(I,4);
        RF(I,139) = RF(I,139)*C(I,4);
        RF(I,140) = RF(I,140)*C(I,4);
        RF(I,141) = RF(I,141)*C(I,7);
        RF(I,142) = RF(I,142)*C(I,8);
        RF(I,144) = RF(I,144)*C(I,9);
        RF(I,145) = RF(I,145)*C(I,9);
        RF(I,146) = RF(I,146)*C(I,9);
        RF(I,148) = RF(I,148)*C(I,2);
        RF(I,149) = RF(I,149)*C(I,2);
        RF(I,150) = RF(I,150)*C(I,2);
        RF(I,151) = RF(I,151)*C(I,3);
        RF(I,152) = RF(I,152)*C(I,5);
        RF(I,153) = RF(I,153)*C(I,4);
        RF(I,154) = RF(I,154)*C(I,4);
        RF(I,155) = RF(I,155)*C(I,15);
        RF(I,156) = RF(I,156)*C(I,15)*C(I,2);
        RF(I,157) = RF(I,157)*C(I,15)*C(I,2);
        RF(I,158) = RF(I,158)*C(I,15)*C(I,3);
        RF(I,159) = RF(I,159)*C(I,15)*C(I,3);
        RF(I,160) = RF(I,160)*C(I,15)*C(I,3);
        RF(I,161) = RF(I,161)*C(I,15)*C(I,5);
        RF(I,162) = RF(I,162)*C(I,15)*C(I,4);
        RF(I,163) = RF(I,163)*C(I,15)*C(I,7);
        RF(I,164) = RF(I,164)*C(I,15);
        RF(I,165) = RF(I,165)*C(I,15);
        RF(I,166) = RF(I,166)*C(I,15);
        RF(I,167) = RF(I,167)*C(I,15);
        RF(I,168) = RF(I,168)*C(I,15)*C(I,9);
        RF(I,169) = RF(I,169)*C(I,15)*C(I,9);
        RF(I,170) = RF(I,170)*C(I,2);
        RF(I,171) = RF(I,171)*C(I,2);
        RF(I,172) = RF(I,172)*C(I,3);
        RF(I,173) = RF(I,173)*C(I,3);
        RF(I,174) = RF(I,174)*C(I,4);
        RF(I,175) = RF(I,175)*C(I,7);
        RF(I,176) = RF(I,176)*C(I,7);
        RF(I,177) = RF(I,177)*C(I,7);
        RF(I,178) = RF(I,178)*C(I,8);
        RF(I,180) = RF(I,180)*C(I,16)*C(I,2);
        RF(I,181) = RF(I,181)*C(I,16)*C(I,3);
        RF(I,182) = RF(I,182)*C(I,16)*C(I,5);
        RF(I,183) = RF(I,183)*C(I,16);
        RF(I,184) = RF(I,184)*C(I,16)*C(I,9);
        RF(I,185) = RF(I,185)*C(I,20)*C(I,2);
        RF(I,186) = RF(I,186)*C(I,20)*C(I,2);
        RF(I,187) = RF(I,187)*C(I,20)*C(I,7);
        RF(I,188) = RF(I,188)*C(I,20)*C(I,7);
        RF(I,189) = RF(I,189)*C(I,20);
        RF(I,190) = RF(I,190)*C(I,21)*C(I,2);
        RF(I,191) = RF(I,191)*C(I,21)*C(I,2);
        RF(I,192) = RF(I,192)*C(I,21)*C(I,2);
        RF(I,193) = RF(I,193)*C(I,21)*C(I,3);
        RF(I,194) = RF(I,194)*C(I,21)*C(I,3);
        RF(I,195) = RF(I,195)*C(I,21)*C(I,3);
        RF(I,196) = RF(I,196)*C(I,21)*C(I,5);
        RF(I,197) = RF(I,197)*C(I,21)*C(I,7);
        RF(I,198) = RF(I,198)*C(I,21)*C(I,9);
        RF(I,199) = RF(I,199)*C(I,2);
        RF(I,200) = RF(I,200)*C(I,2);
        RF(I,201) = RF(I,201)*C(I,3);
        RF(I,202) = RF(I,202)*C(I,5);
        RF(I,203) = RF(I,203)*C(I,4);
        RF(I,204) = RF(I,204)*C(I,7);
        RF(I,205) = RF(I,205)*C(I,9);
    }

    for (I=1; I<=VL; I++) 
    {
        RB(I,1) = RB(I,1)*C(I,3)*C(I,5);
        RB(I,2) = RB(I,2)*C(I,2)*C(I,5);
        RB(I,3) = RB(I,3)*C(I,2)*C(I,6);
        RB(I,4) = RB(I,4)*C(I,3)*C(I,6);
        RB(I,5) = RB(I,5)*CTB(I,5)*C(I,1);
        RB(I,6) = RB(I,6)*C(I,1)*C(I,1);
        RB(I,7) = RB(I,7)*C(I,1)*C(I,6);
        RB(I,8) = RB(I,8)*C(I,1)*C(I,12);
        RB(I,9) = RB(I,9)*CTB(I,9)*C(I,6);
        RB(I,10) = RB(I,10)*CTB(I,10)*C(I,5);
        RB(I,11) = RB(I,11)*CTB(I,11)*C(I,4);
        RB(I,12) = RB(I,12)*CTB(I,12)*C(I,7);
        RB(I,13) = RB(I,13)*C(I,7)*C(I,4);
        RB(I,14) = RB(I,14)*C(I,7)*C(I,6);
        RB(I,15) = RB(I,15)*C(I,7)*C(I,22);
        RB(I,16) = RB(I,16)*C(I,8);
        RB(I,17) = RB(I,17)*C(I,3)*C(I,6);
        RB(I,18) = RB(I,18)*C(I,4)*C(I,1);
        RB(I,19) = RB(I,19)*C(I,5)*C(I,5);
        RB(I,20) = RB(I,20)*C(I,5)*C(I,4);
        RB(I,21) = RB(I,21)*C(I,4)*C(I,6);
        RB(I,22) = RB(I,22)*C(I,4)*C(I,8);
        RB(I,23) = RB(I,23)*C(I,4)*C(I,8);
        RB(I,24) = RB(I,24)*C(I,7)*C(I,1);
        RB(I,25) = RB(I,25)*C(I,5)*C(I,6);
        RB(I,26) = RB(I,26)*C(I,5)*C(I,7);
        RB(I,27) = RB(I,27)*C(I,7)*C(I,6);
        RB(I,28) = RB(I,28)*C(I,7)*C(I,6);
        RB(I,29) = RB(I,29)*CTB(I,29)*C(I,12);
        RB(I,30) = RB(I,30)*C(I,12)*C(I,2);
        RB(I,31) = RB(I,31)*C(I,13);
        RB(I,32) = RB(I,32)*C(I,12)*C(I,3);
        RB(I,33) = RB(I,33)*C(I,12)*C(I,5);
        RB(I,34) = RB(I,34)*C(I,11)*C(I,2);
        RB(I,35) = RB(I,35)*C(I,2);
        RB(I,36) = RB(I,36)*C(I,2);
        RB(I,37) = RB(I,37)*C(I,13)*C(I,2);
        RB(I,38) = RB(I,38)*C(I,3);
        RB(I,39) = RB(I,39)*C(I,17);
        RB(I,40) = RB(I,40)*C(I,11);
        RB(I,41) = RB(I,41)*C(I,13);
        RB(I,42) = RB(I,42)*C(I,11)*C(I,1);
        RB(I,43) = RB(I,43)*C(I,11)*C(I,5);
        RB(I,44) = RB(I,44)*C(I,12)*C(I,2);
        RB(I,45) = RB(I,45)*C(I,11)*C(I,6);
        RB(I,46) = RB(I,46)*CTB(I,46)*C(I,11)*C(I,2);
        RB(I,47) = RB(I,47)*C(I,11)*C(I,7);
        RB(I,48) = RB(I,48)*C(I,9);
        RB(I,49) = RB(I,49)*C(I,2)*C(I,9);
        RB(I,50) = RB(I,50)*C(I,2);
        RB(I,51) = RB(I,51)*C(I,5);
        RB(I,52) = RB(I,52)*C(I,12)*C(I,2)*C(I,2);
        RB(I,53) = RB(I,53)*C(I,13)*C(I,2);
        RB(I,54) = RB(I,54)*C(I,6);
        RB(I,55) = RB(I,55)*C(I,13)*C(I,5);
        RB(I,56) = RB(I,56)*C(I,18);
        RB(I,57) = RB(I,57)*C(I,14)*C(I,2);
        RB(I,58) = RB(I,58)*C(I,14)*C(I,1);
        RB(I,59) = RB(I,59)*C(I,22);
        RB(I,60) = RB(I,60)*C(I,1);
        RB(I,61) = RB(I,61)*C(I,11)*C(I,1);
        RB(I,62) = RB(I,62)*C(I,2);
        RB(I,63) = RB(I,63)*C(I,13)*C(I,2);
        RB(I,64) = RB(I,64)*C(I,9)*C(I,2);
        RB(I,65) = RB(I,65)*C(I,2)*C(I,5)*C(I,11);
        RB(I,66) = RB(I,66)*C(I,11)*C(I,6);
        RB(I,67) = RB(I,67)*C(I,6);
        RB(I,68) = RB(I,68)*C(I,11);
        RB(I,69) = RB(I,69)*C(I,12);
        RB(I,70) = RB(I,70)*C(I,13)*C(I,11);
        RB(I,72) = RB(I,72)*C(I,1);
        RB(I,73) = RB(I,73)*C(I,5);
        RB(I,74) = RB(I,74)*C(I,6);
        RB(I,75) = RB(I,75)*C(I,7);
        RB(I,76) = RB(I,76)*C(I,8);
        RB(I,77) = RB(I,77)*C(I,18)*C(I,2);
        RB(I,78) = RB(I,78)*C(I,10);
        RB(I,79) = RB(I,79)*C(I,13)*C(I,2);
        RB(I,80) = RB(I,80)*C(I,6);
        RB(I,81) = RB(I,81)*C(I,6);
        RB(I,82) = RB(I,82)*C(I,3);
        RB(I,83) = RB(I,83)*C(I,5)*C(I,13);
        RB(I,84) = RB(I,84)*C(I,10)*C(I,4);
        RB(I,85) = RB(I,85)*C(I,5);
        RB(I,86) = RB(I,86)*C(I,10)*C(I,7);
        RB(I,87) = RB(I,87)*C(I,2);
        RB(I,88) = RB(I,88)*C(I,10)*C(I,11);
        RB(I,89) = RB(I,89)*C(I,19);
        RB(I,90) = RB(I,90)*C(I,10);
        RB(I,91) = RB(I,91)*C(I,15)*C(I,2);
        RB(I,92) = RB(I,92)*C(I,15)*C(I,2);
        RB(I,93) = RB(I,93)*C(I,16);
        RB(I,94) = RB(I,94)*C(I,2);
        RB(I,95) = RB(I,95)*C(I,15)*C(I,11);
        RB(I,96) = RB(I,96)*C(I,13)*C(I,1);
        RB(I,97) = RB(I,97)*C(I,9)*C(I,5);
        RB(I,98) = RB(I,98)*C(I,6);
        RB(I,99) = RB(I,99)*C(I,13)*C(I,5);
        RB(I,100) = RB(I,100)*C(I,13)*C(I,6);
        RB(I,101) = RB(I,101)*C(I,13)*C(I,7);
        RB(I,102) = RB(I,102)*C(I,9)*C(I,1);
        RB(I,103) = RB(I,103)*C(I,9)*C(I,5);
        RB(I,104) = RB(I,104)*C(I,9)*C(I,6);
        RB(I,105) = RB(I,105)*C(I,15)*C(I,2);
        RB(I,106) = RB(I,106)*C(I,9)*C(I,9);
        RB(I,107) = RB(I,107)*C(I,9)*C(I,9);
        RB(I,108) = RB(I,108)*C(I,11);
        RB(I,109) = RB(I,109)*C(I,2)*C(I,11)*C(I,11);
        RB(I,110) = RB(I,110)*C(I,5)*C(I,11)*C(I,11);
        RB(I,111) = RB(I,111)*C(I,14)*C(I,11);
        RB(I,112) = RB(I,112)*C(I,11);
        RB(I,113) = RB(I,113)*C(I,14)*C(I,11)*C(I,11);
        RB(I,115) = RB(I,115)*C(I,14)*C(I,2);
        RB(I,116) = RB(I,116)*C(I,17)*C(I,2);
        RB(I,117) = RB(I,117)*C(I,11);
        RB(I,118) = RB(I,118)*C(I,18)*C(I,2);
        RB(I,119) = RB(I,119)*C(I,9)*C(I,11);
        RB(I,120) = RB(I,120)*C(I,11);
        RB(I,121) = RB(I,121)*CTB(I,121)*C(I,20);
        RB(I,122) = RB(I,122)*C(I,14)*C(I,2);
        RB(I,123) = RB(I,123)*C(I,11);
        RB(I,124) = RB(I,124)*C(I,18)*C(I,2);
        RB(I,125) = RB(I,125)*C(I,12);
        RB(I,127) = RB(I,127)*C(I,17)*C(I,1);
        RB(I,128) = RB(I,128)*C(I,9)*C(I,11);
        RB(I,129) = RB(I,129)*C(I,17)*C(I,5);
        RB(I,130) = RB(I,130)*C(I,12);
        RB(I,131) = RB(I,131)*C(I,17)*C(I,6);
        RB(I,132) = RB(I,132)*C(I,15);
        RB(I,133) = RB(I,133)*C(I,14)*C(I,1);
        RB(I,134) = RB(I,134)*C(I,1);
        RB(I,135) = RB(I,135)*C(I,18)*C(I,2);
        RB(I,136) = RB(I,136)*C(I,9)*C(I,11);
        RB(I,137) = RB(I,137)*C(I,14)*C(I,6);
        RB(I,138) = RB(I,138)*C(I,14)*C(I,7);
        RB(I,139) = RB(I,139)*C(I,3);
        RB(I,140) = RB(I,140)*C(I,13);
        RB(I,141) = RB(I,141)*C(I,5);
        RB(I,142) = RB(I,142)*C(I,15)*C(I,7);
        RB(I,143) = RB(I,143)*C(I,15)*C(I,11);
        RB(I,144) = RB(I,144)*C(I,14)*C(I,10);
        RB(I,145) = RB(I,145)*C(I,21);
        RB(I,146) = RB(I,146)*C(I,20)*C(I,2);
        RB(I,147) = RB(I,147)*C(I,9)*C(I,11);
        RB(I,148) = RB(I,148)*C(I,19);
        RB(I,149) = RB(I,149)*C(I,9);
        RB(I,150) = RB(I,150)*C(I,18)*C(I,1);
        RB(I,151) = RB(I,151)*C(I,18)*C(I,5);
        RB(I,152) = RB(I,152)*C(I,18)*C(I,6);
        RB(I,153) = RB(I,153)*C(I,18)*C(I,7);
        RB(I,154) = RB(I,154)*C(I,13)*C(I,11)*C(I,5);
        RB(I,155) = RB(I,155)*C(I,1);
        RB(I,157) = RB(I,157)*C(I,1);
        RB(I,158) = RB(I,158)*C(I,5);
        RB(I,159) = RB(I,159)*C(I,9);
        RB(I,160) = RB(I,160)*C(I,13);
        RB(I,161) = RB(I,161)*C(I,6);
        RB(I,162) = RB(I,162)*C(I,7);
        RB(I,163) = RB(I,163)*C(I,19)*C(I,5);
        RB(I,164) = RB(I,164)*C(I,11);
        RB(I,165) = RB(I,165)*C(I,20)*C(I,2);
        RB(I,166) = RB(I,166)*C(I,10);
        RB(I,167) = RB(I,167)*C(I,20)*C(I,2);
        RB(I,168) = RB(I,168)*C(I,10);
        RB(I,170) = RB(I,170)*C(I,16);
        RB(I,171) = RB(I,171)*C(I,15)*C(I,1);
        RB(I,172) = RB(I,172)*C(I,9)*C(I,13);
        RB(I,173) = RB(I,173)*C(I,19)*C(I,2);
        RB(I,174) = RB(I,174)*C(I,15)*C(I,7);
        RB(I,175) = RB(I,175)*C(I,16)*C(I,4);
        RB(I,176) = RB(I,176)*C(I,15)*C(I,8);
        RB(I,177) = RB(I,177)*C(I,9)*C(I,13)*C(I,5);
        RB(I,178) = RB(I,178)*C(I,16)*C(I,7);
        RB(I,179) = RB(I,179)*C(I,16)*C(I,11);
        RB(I,180) = RB(I,180)*C(I,1);
        RB(I,181) = RB(I,181)*C(I,5);
        RB(I,182) = RB(I,182)*C(I,6);
        RB(I,183) = RB(I,183)*C(I,9);
        RB(I,184) = RB(I,184)*C(I,10);
        RB(I,185) = RB(I,185)*C(I,21);
        RB(I,186) = RB(I,186)*C(I,10);
        RB(I,187) = RB(I,187)*C(I,21)*C(I,4);
        RB(I,188) = RB(I,188)*C(I,5)*C(I,13);
        RB(I,189) = RB(I,189)*C(I,21)*C(I,11);
        RB(I,191) = RB(I,191)*C(I,15)*C(I,9);
        RB(I,192) = RB(I,192)*C(I,20)*C(I,1);
        RB(I,193) = RB(I,193)*C(I,18)*C(I,9)*C(I,2);
        RB(I,195) = RB(I,195)*C(I,20)*C(I,5);
        RB(I,196) = RB(I,196)*C(I,20)*C(I,6);
        RB(I,197) = RB(I,197)*C(I,20)*C(I,8);
        RB(I,198) = RB(I,198)*C(I,20)*C(I,10);
        RB(I,199) = RB(I,199)*C(I,9);
        RB(I,200) = RB(I,200)*C(I,21)*C(I,1);
        RB(I,201) = RB(I,201)*C(I,13);
        RB(I,202) = RB(I,202)*C(I,21)*C(I,6);
        RB(I,203) = RB(I,203)*C(I,21)*C(I,7);
        RB(I,204) = RB(I,204)*C(I,5)*C(I,13);
        RB(I,205) = RB(I,205)*C(I,10)*C(I,21);
        RB(I,206) = RB(I,206)*C(I,20)*C(I,9);
    }
}
