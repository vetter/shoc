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
ratt_i_VEC(real * RESTRICT T, real * RESTRICT RF, real * RESTRICT RB, 
        real * RESTRICT RKLOW) 
{

    const real RU=8.314510e7, RUC=RU/4.184e7, PATM=1.01325e6;
    const real SMALL = floatMin<real>(); //1.e-300;
    real PFAC, PFAC2, PFAC3;
    real PFACI;
    ALIGN64 real SMH[MAXVL*32], EG[MAXVL*32], EQK[MAXVL*206], EGI[MAXVL*32],
            ALOGT[MAXVL],TI[MAXVL], TI2[MAXVL],TMP[MAXVL];
    int VL = MAXVL;
    int J,I, K, N, VL31;

    vrda_log_(&VL,T,ALOGT);


    #pragma vector aligned
    for (I=1; I<=VL; I++) TI(I) = 1.0e0/T(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) TI2(I) = TI(I)*TI(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,1) = EXP(3.20498617e1 -7.25286183e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,2) = EXP(1.08197783e1 +2.67e0*ALOGT(I) -3.16523284e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,3) = EXP(1.9190789e1 +1.51e0*ALOGT(I) -1.72603317e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,4) = EXP(1.0482906e1 +2.4e0*ALOGT(I) +1.06178717e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,5) = 1.e18*TI(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,6) = EXP(3.90385861e1 -6.e-1*ALOGT(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,7) = EXP(4.55408762e1 -1.25e0*ALOGT(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,8) = 5.5e20*TI2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,9) = 2.2e22*TI2(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,10) = 5.e17*TI(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,11) = 1.2e17*TI(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,12) = EXP(4.24761511e1 -8.6e-1*ALOGT(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,13) = EXP(4.71503141e1 -1.72e0*ALOGT(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,14) = EXP(4.42511034e1 -7.6e-1*ALOGT(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,15) = EXP(4.47046282e1 -1.24e0*ALOGT(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,16) = EXP(3.19350862e1 -3.7e-1*ALOGT(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,17) = EXP(2.90097872e1 -3.37658384e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,18) = EXP(3.04404238e1 -4.12637667e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,19) = EXP(3.18908801e1 -1.50965e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,20) = 2.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,21) = EXP(3.14683206e1 +2.51608334e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,22) = EXP(2.55908003e1 +8.20243168e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,23) = EXP(3.36712758e1 -6.03860001e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,24) = EXP(1.6308716e1 +2.e0*ALOGT(I) -2.61672667e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,25) = EXP(2.99336062e1 -1.81158e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,26) = EXP(1.60803938e1 +2.e0*ALOGT(I) -2.01286667e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,27) = EXP(2.81906369e1 -1.61029334e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,28) = EXP(3.39940492e1 -4.81075134e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,29) = EXP(3.40312786e1 -1.50965e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,30) = EXP(1.76783433e1 +1.228e0*ALOGT(I) -3.52251667e1*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,31) = EXP(1.75767107e1 +1.5e0*ALOGT(I) -4.00560467e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,32) = EXP(2.85473118e1 -2.40537567e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,33) = EXP(3.26416564e1 -1.18759134e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,34) = 5.7e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,35) = 3.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,36) = EXP(1.85223344e1 +1.79e0*ALOGT(I) -8.40371835e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,37) = EXP(2.93732401e1 +3.79928584e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,38) = 3.3e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,39) = 5.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,40) = EXP(2.88547965e1 -3.47219501e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,41) = EXP(2.77171988e1 +4.8e-1*ALOGT(I) +1.30836334e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,42) = 7.34e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,43) = 3.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,44) = 3.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,45) = 5.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,46) = EXP(3.9769885e1 -1.e0*ALOGT(I) -8.55468335e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,47) = EXP(2.96591694e1 -2.01286667e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,48) = EXP(3.77576522e1 -8.e-1*ALOGT(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,49) = EXP(1.31223634e1 +2.e0*ALOGT(I) -3.63825651e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,50) = 8.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) TMP(I) = EXP(-7.54825001e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,51) = 1.056e13 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,52) = 2.64e12 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,53) = 2.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,54) = EXP(1.62403133e1 +2.e0*ALOGT(I) -1.50965e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,55) = 2.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,56) = EXP(2.74203001e1 +5.e-1*ALOGT(I) -2.26950717e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,57) = 4.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,58) = 3.2e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,59) = EXP(3.03390713e1 -3.01930001e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,60) = 3.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,61) = 1.5e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,62) = 1.5e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,63) = 3.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,64) = 7.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,65) = 2.8e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,66) = 1.2e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,67) = 3.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,68) = 9.e12;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,69) = 7.e12;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,70) = 1.4e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,71) = EXP(2.7014835e1 +4.54e-1*ALOGT(I) -1.30836334e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,72) = EXP(2.38587601e1 +1.05e0*ALOGT(I) -1.64803459e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,73) = EXP(3.12945828e1 -1.781387e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,74) = EXP(2.19558261e1 +1.18e0*ALOGT(I) +2.2493785e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,75) = EXP(3.22361913e1 -2.01286667e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) TMP(I) = EXP(-4.02573334e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,76) = 1.e12 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,127) = 5.e13 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,129) = 1.e13 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,77) = EXP(3.21806786e1 +2.59156584e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,78) = EXP(3.70803784e1 -6.3e-1*ALOGT(I) -1.92731984e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,79) = 8.43e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,80) = EXP(1.78408622e1 +1.6e0*ALOGT(I) -2.72743434e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,81) = 2.501e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,82) = EXP(3.10595094e1 -1.449264e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,83) = EXP(2.43067848e1 -4.49875701e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,84) = 1.e12;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,85) = 1.34e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,86) = EXP(1.01064284e1 +2.47e0*ALOGT(I) -2.60666234e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,87) = 3.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,88) = 8.48e12;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,89) = 1.8e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,90) = EXP(8.10772006e0 +2.81e0*ALOGT(I) -2.94884967e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,91) = 4.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) TMP(I) = EXP(2.86833501e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,92) = 1.2e13 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,107) = 1.6e13 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,93) = EXP(3.75927776e1 -9.7e-1*ALOGT(I) -3.11994334e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,94) = EXP(2.9238457e1 +1.e-1*ALOGT(I) -5.33409668e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,95) = 5.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,96) = 2.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,97) = 3.2e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,98) = 1.6e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,99) = 1.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,100) = 5.e12;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,101) = EXP(-2.84796532e1 +7.6e0*ALOGT(I) +1.77635484e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,102) = EXP(2.03077504e1 +1.62e0*ALOGT(I) -5.45486868e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,103) = EXP(2.07430685e1 +1.5e0*ALOGT(I) -4.32766334e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,104) = EXP(1.84206807e1 +1.6e0*ALOGT(I) -1.570036e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,105) = 6.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,106) = EXP(1.47156719e1 +2.e0*ALOGT(I) -4.16160184e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,108) = 1.e14;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,109) = 1.e14;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,110) = EXP(2.81010247e1 -4.29747034e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,111) = 5.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,112) = 3.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,113) = 1.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,114) = EXP(3.43156328e1 -5.2e-1*ALOGT(I) -2.55382459e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,115) = EXP(1.97713479e1 +1.62e0*ALOGT(I) -1.86432818e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) TMP(I) = EXP(2.e0*ALOGT(I) -9.56111669e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,116) = 1.632e7 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,117) = 4.08e6 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,118) = EXP(-8.4310155e0 +4.5e0*ALOGT(I) +5.03216668e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,119) = EXP(-7.6354939e0 +4.e0*ALOGT(I) +1.00643334e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,120) = EXP(1.61180957e1 +2.e0*ALOGT(I) -3.01930001e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,121) = EXP(1.27430637e2 -1.182e1*ALOGT(I) -1.79799315e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,122) = 1.e14;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,123) = 1.e14;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,124) = 2.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,125) = 1.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,126) = EXP(3.34301138e1 -6.e-2*ALOGT(I) -4.27734167e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,128) = EXP(2.11287309e1 +1.43e0*ALOGT(I) -1.35365284e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,130) = EXP(2.81906369e1 -6.79342501e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) TMP(I) = EXP(-1.00643334e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,131) = 7.5e12 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,152) = 1.e13 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,186) = 2.e13 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,132) = EXP(2.94360258e1 +2.7e-1*ALOGT(I) -1.40900667e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,133) = 3.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,134) = 6.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,135) = 4.8e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,136) = 4.8e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,137) = 3.011e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,138) = EXP(1.41081802e1 +1.61e0*ALOGT(I) +1.9293327e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,139) = EXP(2.64270483e1 +2.9e-1*ALOGT(I) -5.53538334e0*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,140) = EXP(3.83674178e1 -1.39e0*ALOGT(I) -5.08248834e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,141) = 1.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,142) = EXP(2.32164713e1 +2.99917134e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,143) = 9.033e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,144) = 3.92e11;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,145) = 2.5e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,146) = EXP(5.56675073e1 -2.83e0*ALOGT(I) -9.36888792e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,147) = EXP(9.64601125e1 -9.147e0*ALOGT(I) -2.36008617e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,148) = 1.e14;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,149) = 9.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) TMP(I) = EXP(-2.01286667e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,150) = 2.e13 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,151) = 2.e13 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,153) = 1.4e11;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,154) = 1.8e10;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,155) = EXP(2.97104627e1 +4.4e-1*ALOGT(I) -4.46705436e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,156) = EXP(2.77079822e1 +4.54e-1*ALOGT(I) -9.15854335e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,157) = EXP(1.77414365e1 +1.93e0*ALOGT(I) -6.51665585e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,158) = EXP(1.65302053e1 +1.91e0*ALOGT(I) -1.88203034e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) TMP(I) = EXP(1.83e0*ALOGT(I) -1.10707667e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,159) = 1.92e7 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,160) = 3.84e5 * TMP(I);
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,161) = EXP(1.50964444e1 +2.e0*ALOGT(I) -1.25804167e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,162) = EXP(3.13734413e1 -3.05955734e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,163) = EXP(2.83241683e1 -7.04503335e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,164) = EXP(1.61180957e1 +2.e0*ALOGT(I) -4.02573334e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,165) = EXP(3.06267534e1 -3.01930001e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,166) = 5.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,167) = 5.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,168) = EXP(1.23327053e1 +2.e0*ALOGT(I) -4.62959334e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,169) = EXP(2.65223585e1 -3.87476834e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,170) = EXP(4.07945264e1 -9.9e-1*ALOGT(I) -7.95082335e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,171) = 2.e12;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,172) = 1.604e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,173) = 8.02e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,174) = 2.e10;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,175) = 3.e11;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,176) = 3.e11;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,177) = 2.4e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,178) = EXP(2.28865889e1 -4.90133034e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,179) = 1.2e14;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,180) = EXP(1.85604427e1 +1.9e0*ALOGT(I) -3.78922151e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,181) = EXP(1.83130955e1 +1.92e0*ALOGT(I) -2.86330284e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,182) = EXP(1.50796373e1 +2.12e0*ALOGT(I) -4.37798501e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,183) = EXP(3.13199006e1 +2.76769167e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,184) = EXP(1.56303353e1 +1.74e0*ALOGT(I) -5.25861418e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,185) = 2.e14;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,187) = 2.66e12;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,188) = 6.6e12;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,189) = 6.e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,190) = EXP(3.02187852e1 -1.64083859e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,191) = EXP(5.11268757e1 -2.39e0*ALOGT(I) -5.62596234e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,192) = EXP(1.20435537e1 +2.5e0*ALOGT(I) -1.2530095e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,193) = EXP(1.86030023e1 +1.65e0*ALOGT(I) -1.6455185e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,194) = EXP(1.73708586e1 +1.65e0*ALOGT(I) +4.89126601e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,195) = EXP(2.59162227e1 +7.e-1*ALOGT(I) -2.95891401e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,196) = EXP(1.49469127e1 +2.e0*ALOGT(I) +1.49958567e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,197) = EXP(9.16951838e0 +2.6e0*ALOGT(I) -6.99974385e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,198) = EXP(7.8845736e-1 +3.5e0*ALOGT(I) -2.85575459e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,199) = EXP(5.65703751e1 -2.92e0*ALOGT(I) -6.29272443e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,200) = 1.8e12;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,201) = 9.6e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,202) = 2.4e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,203) = 9.e10;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,204) = 2.4e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,205) = 1.1e13;
    #pragma vector aligned
    for (I=1; I<=VL; I++) RF(I,206) = EXP(7.50436995e1 -5.22e0*ALOGT(I) -9.93701954e3*TI(I));

    rdsmh_i_VEC<real,MAXVL>(T, SMH);
    if(VL==MAXVL) {
        VL31 = 31*MAXVL;
        vrda_exp_(&VL31,&(SMH(1,1)),&(EG(1,1)));
    }
    else {
    #pragma vector aligned
        for (N = 1; N<= 31; N++) {
            for (I=0; I<VL; I++) EG[MAXVL*(N-1)+I] = EXP(SMH[MAXVL*(N-1)+I]);        }
    }

    for (J=1; J<=31; J++) 
    #pragma vector aligned
        for (I=1; I<=VL; I++)
            EGI(I,J)=1.0/EG(I,J);

    #pragma vector aligned
    for (I=1; I<=VL; I++) {
        PFAC = PATM / (RU*T(I));
        PFACI = (RU*T(I))/PATM;
        PFAC2 = PFAC*PFAC;
        PFAC3 = PFAC2*PFAC;

        EQK(I,1)=EG(I,3)*EG(I,5)*EGI(I,2)*EGI(I,4);
        EQK(I,2)=EG(I,2)*EG(I,5)*EGI(I,1)*EGI(I,3);
        EQK(I,3)=EG(I,2)*EG(I,6)*EGI(I,1)*EGI(I,5);
        EQK(I,4)=EG(I,3)*EG(I,6)*EGI(I,5)*EGI(I,5);
        EQK(I,5)=EG(I,1)*EGI(I,2)*EGI(I,2)*PFACI;
        EQK(I,6)=EQK(I,5);
        EQK(I,7)=EQK(I,5);
        EQK(I,8)=EQK(I,5);
        EQK(I,9)=EG(I,6)*EGI(I,2)*EGI(I,5)*PFACI;
        EQK(I,10)=EG(I,5)*EGI(I,2)*EGI(I,3)*PFACI;
        EQK(I,11)=EG(I,4)*EGI(I,3)*EGI(I,3)*PFACI;
        EQK(I,12)=EG(I,7)*EGI(I,2)*EGI(I,4)*PFACI;
        EQK(I,13)=EQK(I,12);
        EQK(I,14)=EQK(I,12);
        EQK(I,15)=EQK(I,12);
        EQK(I,16)=EG(I,8)*EGI(I,5)*EGI(I,5)*PFACI;
        EQK(I,17)=EG(I,3)*EG(I,6)*EGI(I,2)*EGI(I,7);
        EQK(I,18)=EG(I,1)*EG(I,4)*EGI(I,2)*EGI(I,7);
        EQK(I,19)=EG(I,5)*EG(I,5)*EGI(I,2)*EGI(I,7);
        EQK(I,20)=EG(I,4)*EG(I,5)*EGI(I,3)*EGI(I,7);
        EQK(I,21)=EG(I,4)*EG(I,6)*EGI(I,5)*EGI(I,7);
        EQK(I,22)=EG(I,4)*EG(I,8)*EGI(I,7)*EGI(I,7);
        EQK(I,23)=EQK(I,22);
        EQK(I,24)=EG(I,1)*EG(I,7)*EGI(I,2)*EGI(I,8);
        EQK(I,25)=EG(I,5)*EG(I,6)*EGI(I,2)*EGI(I,8);
        EQK(I,26)=EG(I,5)*EG(I,7)*EGI(I,3)*EGI(I,8);
        EQK(I,27)=EG(I,6)*EG(I,7)*EGI(I,5)*EGI(I,8);
        EQK(I,28)=EQK(I,27);
        EQK(I,29)=EG(I,15)*EGI(I,3)*EGI(I,14)*PFACI;
        EQK(I,30)=EG(I,2)*EG(I,15)*EGI(I,5)*EGI(I,14);
        EQK(I,31)=EG(I,17)*EGI(I,1)*EGI(I,14)*PFACI;
        EQK(I,32)=EG(I,3)*EG(I,15)*EGI(I,4)*EGI(I,14);
        EQK(I,33)=EG(I,5)*EG(I,15)*EGI(I,7)*EGI(I,14);
        EQK(I,34)=EG(I,2)*EG(I,14)*EGI(I,3)*EGI(I,9);
        EQK(I,35)=EG(I,2)*EG(I,16)*EGI(I,5)*EGI(I,9);
        EQK(I,36)=EG(I,2)*EG(I,10)*EGI(I,1)*EGI(I,9);
        EQK(I,37)=EG(I,2)*EG(I,17)*EGI(I,6)*EGI(I,9);
        EQK(I,38)=EG(I,3)*EG(I,16)*EGI(I,4)*EGI(I,9);
        EQK(I,39)=EG(I,25)*EGI(I,9)*EGI(I,14)*PFACI;
        EQK(I,40)=EG(I,14)*EG(I,16)*EGI(I,9)*EGI(I,15);
        EQK(I,41)=EG(I,17)*EGI(I,2)*EGI(I,16)*PFACI;
        EQK(I,42)=EG(I,1)*EG(I,14)*EGI(I,2)*EGI(I,16);
        EQK(I,43)=EG(I,5)*EG(I,14)*EGI(I,3)*EGI(I,16);
        EQK(I,44)=EG(I,2)*EG(I,15)*EGI(I,3)*EGI(I,16);
        EQK(I,45)=EG(I,6)*EG(I,14)*EGI(I,5)*EGI(I,16);
        EQK(I,46)=EG(I,2)*EG(I,14)*EGI(I,16)*PFAC;
        EQK(I,47)=EG(I,7)*EG(I,14)*EGI(I,4)*EGI(I,16);
        EQK(I,48)=EG(I,12)*EGI(I,2)*EGI(I,10)*PFACI;
        EQK(I,49)=EG(I,2)*EG(I,12)*EGI(I,1)*EGI(I,10);
        EQK(I,50)=EG(I,2)*EG(I,16)*EGI(I,3)*EGI(I,10);
        EQK(I,51)=EG(I,5)*EG(I,16)*EGI(I,4)*EGI(I,10);
        EQK(I,52)=EG(I,2)*EG(I,2)*EG(I,15)*EGI(I,4)*EGI(I,10)*PFAC;
        EQK(I,53)=EG(I,2)*EG(I,17)*EGI(I,5)*EGI(I,10);
        EQK(I,54)=EG(I,6)*EG(I,9)*EGI(I,5)*EGI(I,10);
        EQK(I,55)=EG(I,5)*EG(I,17)*EGI(I,7)*EGI(I,10);
        EQK(I,56)=EG(I,26)*EGI(I,10)*EGI(I,14)*PFACI;
        EQK(I,57)=EG(I,2)*EG(I,19)*EGI(I,9)*EGI(I,10);
        EQK(I,58)=EG(I,1)*EG(I,19)*EGI(I,10)*EGI(I,10);
        EQK(I,59)=EG(I,10)*EGI(I,11);
        EQK(I,67)=EQK(I,59);
        EQK(I,68)=EQK(I,59);
        EQK(I,69)=EQK(I,59);
        EQK(I,60)=EG(I,1)*EG(I,9)*EGI(I,2)*EGI(I,11);
        EQK(I,61)=EG(I,1)*EG(I,14)*EGI(I,3)*EGI(I,11);
        EQK(I,62)=EG(I,2)*EG(I,16)*EGI(I,3)*EGI(I,11);
        EQK(I,63)=EG(I,2)*EG(I,17)*EGI(I,5)*EGI(I,11);
        EQK(I,64)=EG(I,2)*EG(I,12)*EGI(I,1)*EGI(I,11);
        EQK(I,65)=EG(I,2)*EG(I,5)*EG(I,14)*EGI(I,4)*EGI(I,11)*PFAC;
        EQK(I,66)=EG(I,6)*EG(I,14)*EGI(I,4)*EGI(I,11);
        EQK(I,70)=EG(I,14)*EG(I,17)*EGI(I,11)*EGI(I,15);
        EQK(I,71)=EG(I,18)*EGI(I,2)*EGI(I,17)*PFACI;
        EQK(I,72)=EG(I,1)*EG(I,16)*EGI(I,2)*EGI(I,17);
        EQK(I,73)=EG(I,5)*EG(I,16)*EGI(I,3)*EGI(I,17);
        EQK(I,74)=EG(I,6)*EG(I,16)*EGI(I,5)*EGI(I,17);
        EQK(I,75)=EG(I,7)*EG(I,16)*EGI(I,4)*EGI(I,17);
        EQK(I,76)=EG(I,8)*EG(I,16)*EGI(I,7)*EGI(I,17);
        EQK(I,77)=EG(I,2)*EG(I,26)*EGI(I,9)*EGI(I,17);
        EQK(I,78)=EG(I,13)*EGI(I,2)*EGI(I,12)*PFACI;
        EQK(I,79)=EG(I,2)*EG(I,17)*EGI(I,3)*EGI(I,12);
        EQK(I,80)=EG(I,6)*EG(I,10)*EGI(I,5)*EGI(I,12);
        EQK(I,81)=EG(I,6)*EG(I,11)*EGI(I,5)*EGI(I,12);
        EQK(I,82)=EG(I,3)*EG(I,18)*EGI(I,4)*EGI(I,12);
        EQK(I,83)=EG(I,5)*EG(I,17)*EGI(I,4)*EGI(I,12);
        EQK(I,84)=EG(I,4)*EG(I,13)*EGI(I,7)*EGI(I,12);
        EQK(I,85)=EG(I,5)*EG(I,18)*EGI(I,7)*EGI(I,12);
        EQK(I,86)=EG(I,7)*EG(I,13)*EGI(I,8)*EGI(I,12);
        EQK(I,87)=EG(I,2)*EG(I,21)*EGI(I,9)*EGI(I,12);
        EQK(I,88)=EG(I,13)*EG(I,14)*EGI(I,12)*EGI(I,16);
        EQK(I,89)=EG(I,28)*EGI(I,12)*EGI(I,16)*PFACI;
        EQK(I,90)=EG(I,13)*EG(I,16)*EGI(I,12)*EGI(I,17);
        EQK(I,91)=EG(I,2)*EG(I,22)*EGI(I,10)*EGI(I,12);
        EQK(I,92)=EG(I,2)*EG(I,22)*EGI(I,11)*EGI(I,12);
        EQK(I,93)=EG(I,24)*EGI(I,12)*EGI(I,12)*PFACI;
        EQK(I,94)=EG(I,2)*EG(I,23)*EGI(I,12)*EGI(I,12);
        EQK(I,95)=EG(I,14)*EG(I,22)*EGI(I,12)*EGI(I,25);
        EQK(I,96)=EG(I,1)*EG(I,17)*EGI(I,2)*EGI(I,18);
        EQK(I,97)=EG(I,5)*EG(I,12)*EGI(I,2)*EGI(I,18);
        EQK(I,98)=EG(I,6)*EG(I,11)*EGI(I,2)*EGI(I,18);
        EQK(I,99)=EG(I,5)*EG(I,17)*EGI(I,3)*EGI(I,18);
        EQK(I,100)=EG(I,6)*EG(I,17)*EGI(I,5)*EGI(I,18);
        EQK(I,101)=EG(I,7)*EG(I,17)*EGI(I,4)*EGI(I,18);
        EQK(I,102)=EG(I,1)*EG(I,12)*EGI(I,2)*EGI(I,13);
        EQK(I,103)=EG(I,5)*EG(I,12)*EGI(I,3)*EGI(I,13);
        EQK(I,104)=EG(I,6)*EG(I,12)*EGI(I,5)*EGI(I,13);
        EQK(I,105)=EG(I,2)*EG(I,22)*EGI(I,9)*EGI(I,13);
        EQK(I,106)=EG(I,12)*EG(I,12)*EGI(I,10)*EGI(I,13);
        EQK(I,107)=EG(I,12)*EG(I,12)*EGI(I,11)*EGI(I,13);
        EQK(I,108)=EG(I,11)*EG(I,14)*EGI(I,2)*EGI(I,25);
        EQK(I,109)=EG(I,2)*EG(I,14)*EG(I,14)*EGI(I,3)*EGI(I,25)*PFAC;
        EQK(I,110)=EG(I,5)*EG(I,14)*EG(I,14)*EGI(I,4)*EGI(I,25)*PFAC;
        EQK(I,111)=EG(I,14)*EG(I,19)*EGI(I,9)*EGI(I,25);
        EQK(I,112)=EG(I,14)*EG(I,21)*EGI(I,10)*EGI(I,25);
        EQK(I,113)=EG(I,14)*EG(I,14)*EG(I,19)*EGI(I,25)*EGI(I,25)*PFAC;
        EQK(I,114)=EG(I,20)*EGI(I,19);
        EQK(I,122)=1.0/EQK(I,114);
        EQK(I,115)=EG(I,2)*EG(I,19)*EGI(I,21)*PFAC;
        EQK(I,116)=EG(I,2)*EG(I,25)*EGI(I,3)*EGI(I,19);
        EQK(I,117)=EG(I,10)*EG(I,14)*EGI(I,3)*EGI(I,19);
        EQK(I,118)=EG(I,2)*EG(I,26)*EGI(I,5)*EGI(I,19);
        EQK(I,119)=EG(I,12)*EG(I,14)*EGI(I,5)*EGI(I,19);
        EQK(I,120)=EG(I,14)*EG(I,21)*EGI(I,16)*EGI(I,19);
        EQK(I,121)=EG(I,29)*EGI(I,12)*EGI(I,19)*PFACI;
        EQK(I,123)=EG(I,10)*EG(I,14)*EGI(I,3)*EGI(I,20);
        EQK(I,124)=EG(I,2)*EG(I,26)*EGI(I,5)*EGI(I,20);
        EQK(I,125)=EG(I,10)*EG(I,15)*EGI(I,4)*EGI(I,20);
        EQK(I,126)=EG(I,27)*EGI(I,2)*EGI(I,26)*PFACI;
        EQK(I,127)=EG(I,1)*EG(I,25)*EGI(I,2)*EGI(I,26);
        EQK(I,128)=EG(I,12)*EG(I,14)*EGI(I,2)*EGI(I,26);
        EQK(I,129)=EG(I,5)*EG(I,25)*EGI(I,3)*EGI(I,26);
        EQK(I,130)=EG(I,10)*EG(I,15)*EGI(I,3)*EGI(I,26);
        EQK(I,131)=EG(I,6)*EG(I,25)*EGI(I,5)*EGI(I,26);
        EQK(I,132)=EG(I,22)*EGI(I,2)*EGI(I,21)*PFACI;
        EQK(I,133)=EG(I,1)*EG(I,19)*EGI(I,2)*EGI(I,21);
        EQK(I,134)=EG(I,1)*EG(I,20)*EGI(I,2)*EGI(I,21);
        EQK(I,135)=EG(I,2)*EG(I,26)*EGI(I,3)*EGI(I,21);
        EQK(I,136)=EG(I,12)*EG(I,14)*EGI(I,3)*EGI(I,21);
        EQK(I,137)=EG(I,6)*EG(I,19)*EGI(I,5)*EGI(I,21);
        EQK(I,138)=EG(I,7)*EG(I,19)*EGI(I,4)*EGI(I,21);
        EQK(I,139)=EG(I,3)*EG(I,27)*EGI(I,4)*EGI(I,21);
        EQK(I,140)=EG(I,16)*EG(I,17)*EGI(I,4)*EGI(I,21);
        EQK(I,141)=EG(I,5)*EG(I,27)*EGI(I,7)*EGI(I,21);
        EQK(I,142)=EG(I,7)*EG(I,22)*EGI(I,8)*EGI(I,21);
        EQK(I,143)=EG(I,14)*EG(I,22)*EGI(I,16)*EGI(I,21);
        EQK(I,144)=EG(I,13)*EG(I,19)*EGI(I,12)*EGI(I,21);
        EQK(I,145)=EG(I,30)*EGI(I,12)*EGI(I,21)*PFACI;
        EQK(I,146)=EG(I,2)*EG(I,29)*EGI(I,12)*EGI(I,21);
        EQK(I,147)=EG(I,12)*EG(I,14)*EGI(I,27)*PFAC;
        EQK(I,148)=EG(I,28)*EGI(I,2)*EGI(I,27)*PFACI;
        EQK(I,149)=EG(I,12)*EG(I,16)*EGI(I,2)*EGI(I,27);
        EQK(I,150)=EG(I,1)*EG(I,26)*EGI(I,2)*EGI(I,27);
        EQK(I,151)=EG(I,5)*EG(I,26)*EGI(I,3)*EGI(I,27);
        EQK(I,152)=EG(I,6)*EG(I,26)*EGI(I,5)*EGI(I,27);
        EQK(I,153)=EG(I,7)*EG(I,26)*EGI(I,4)*EGI(I,27);
        EQK(I,154)=EG(I,5)*EG(I,14)*EG(I,17)*EGI(I,4)*EGI(I,27)*PFAC;
        EQK(I,155)=EG(I,1)*EG(I,20)*EGI(I,22)*PFAC;
        EQK(I,156)=EG(I,23)*EGI(I,2)*EGI(I,22)*PFACI;
        EQK(I,157)=EG(I,1)*EG(I,21)*EGI(I,2)*EGI(I,22);
        EQK(I,158)=EG(I,5)*EG(I,21)*EGI(I,3)*EGI(I,22);
        EQK(I,159)=EG(I,12)*EG(I,16)*EGI(I,3)*EGI(I,22);
        EQK(I,160)=EG(I,10)*EG(I,17)*EGI(I,3)*EGI(I,22);
        EQK(I,161)=EG(I,6)*EG(I,21)*EGI(I,5)*EGI(I,22);
        EQK(I,162)=EG(I,7)*EG(I,21)*EGI(I,4)*EGI(I,22);
        EQK(I,163)=EG(I,5)*EG(I,28)*EGI(I,7)*EGI(I,22);
        EQK(I,164)=EG(I,14)*EG(I,23)*EGI(I,16)*EGI(I,22);
        EQK(I,165)=EG(I,2)*EG(I,29)*EGI(I,10)*EGI(I,22);
        EQK(I,166)=EG(I,13)*EG(I,20)*EGI(I,11)*EGI(I,22);
        EQK(I,167)=EG(I,2)*EG(I,29)*EGI(I,11)*EGI(I,22);
        EQK(I,168)=EG(I,13)*EG(I,21)*EGI(I,12)*EGI(I,22);
        EQK(I,169)=EG(I,31)*EGI(I,12)*EGI(I,22)*PFACI;
        EQK(I,170)=EG(I,24)*EGI(I,2)*EGI(I,23)*PFACI;
        EQK(I,171)=EG(I,1)*EG(I,22)*EGI(I,2)*EGI(I,23);
        EQK(I,172)=EG(I,12)*EG(I,17)*EGI(I,3)*EGI(I,23);
        EQK(I,173)=EG(I,2)*EG(I,28)*EGI(I,3)*EGI(I,23);
        EQK(I,174)=EG(I,7)*EG(I,22)*EGI(I,4)*EGI(I,23);
        EQK(I,175)=EG(I,4)*EG(I,24)*EGI(I,7)*EGI(I,23);
        EQK(I,176)=EG(I,8)*EG(I,22)*EGI(I,7)*EGI(I,23);
        EQK(I,177)=EG(I,5)*EG(I,12)*EG(I,17)*EGI(I,7)*EGI(I,23)*PFAC;
        EQK(I,178)=EG(I,7)*EG(I,24)*EGI(I,8)*EGI(I,23);
        EQK(I,179)=EG(I,14)*EG(I,24)*EGI(I,16)*EGI(I,23);
        EQK(I,180)=EG(I,1)*EG(I,23)*EGI(I,2)*EGI(I,24);
        EQK(I,181)=EG(I,5)*EG(I,23)*EGI(I,3)*EGI(I,24);
        EQK(I,182)=EG(I,6)*EG(I,23)*EGI(I,5)*EGI(I,24);
        EQK(I,183)=EG(I,12)*EG(I,23)*EGI(I,11)*EGI(I,24);
        EQK(I,184)=EG(I,13)*EG(I,23)*EGI(I,12)*EGI(I,24);
        EQK(I,185)=EG(I,30)*EGI(I,2)*EGI(I,29)*PFACI;
        EQK(I,186)=EG(I,13)*EG(I,20)*EGI(I,2)*EGI(I,29);
        EQK(I,187)=EG(I,4)*EG(I,30)*EGI(I,7)*EGI(I,29);
        EQK(I,188)=EG(I,5)*EG(I,17)*EG(I,21)*EGI(I,7)*EGI(I,29)*PFAC;
        EQK(I,189)=EG(I,14)*EG(I,30)*EGI(I,16)*EGI(I,29);
        EQK(I,190)=EG(I,31)*EGI(I,2)*EGI(I,30)*PFACI;
        EQK(I,191)=EG(I,12)*EG(I,22)*EGI(I,2)*EGI(I,30);
        EQK(I,192)=EG(I,1)*EG(I,29)*EGI(I,2)*EGI(I,30);
        EQK(I,193)=EG(I,2)*EG(I,12)*EG(I,26)*EGI(I,3)*EGI(I,30)*PFAC;
        EQK(I,194)=EG(I,16)*EG(I,23)*EGI(I,3)*EGI(I,30);
        EQK(I,195)=EG(I,5)*EG(I,29)*EGI(I,3)*EGI(I,30);
        EQK(I,196)=EG(I,6)*EG(I,29)*EGI(I,5)*EGI(I,30);
        EQK(I,197)=EG(I,8)*EG(I,29)*EGI(I,7)*EGI(I,30);
        EQK(I,198)=EG(I,13)*EG(I,29)*EGI(I,12)*EGI(I,30);
        EQK(I,199)=EG(I,12)*EG(I,23)*EGI(I,2)*EGI(I,31);
        EQK(I,200)=EG(I,1)*EG(I,30)*EGI(I,2)*EGI(I,31);
        EQK(I,201)=EG(I,17)*EG(I,23)*EGI(I,3)*EGI(I,31);
        EQK(I,202)=EG(I,6)*EG(I,30)*EGI(I,5)*EGI(I,31);
        EQK(I,203)=EG(I,7)*EG(I,30)*EGI(I,4)*EGI(I,31);
        EQK(I,204)=EG(I,5)*EG(I,17)*EG(I,23)*EGI(I,7)*EGI(I,31)*PFAC;
        EQK(I,205)=EG(I,13)*EG(I,30)*EGI(I,12)*EGI(I,31);
        EQK(I,206)=EG(I,12)*EG(I,29)*EGI(I,21)*EGI(I,23);
    }


    #pragma vector aligned
    for (I=0; I< 206*VL; I++)
        RB[I] = RF[I]/MAX(EQK[I],SMALL);

    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,1) = EXP(4.22794408e1 -9.e-1*ALOGT(I) +8.55468335e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,2) = EXP(6.37931383e1 -3.42e0*ALOGT(I) -4.24463259e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,3) = EXP(6.54619238e1 -3.74e0*ALOGT(I) -9.74227469e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,4) = EXP(5.55621468e1 -2.57e0*ALOGT(I) -7.17083751e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,5) = EXP(6.33329483e1 -3.14e0*ALOGT(I) -6.18956501e2*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,6) = EXP(7.69748493e1 -5.11e0*ALOGT(I) -3.57032226e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,7) = EXP(6.98660102e1 -4.8e0*ALOGT(I) -2.79788467e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,8) = EXP(7.68923562e1 -4.76e0*ALOGT(I) -1.22784867e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,9) = EXP(1.11312542e2 -9.588e0*ALOGT(I) -2.566405e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,10) = EXP(1.15700234e2 -9.67e0*ALOGT(I) -3.13000767e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,11) = EXP(3.54348644e1 -6.4e-1*ALOGT(I) -2.50098684e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,12) = EXP(6.3111756e1 -3.4e0*ALOGT(I) -1.80145126e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,13) = EXP(9.57409899e1 -7.64e0*ALOGT(I) -5.98827834e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,14) = EXP(6.9414025e1 -3.86e0*ALOGT(I) -1.67067934e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,15) = EXP(1.35001549e2 -1.194e1*ALOGT(I) -4.9163262e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,16) = EXP(9.14494773e1 -7.297e0*ALOGT(I) -2.36511834e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,17) = EXP(1.17075165e2 -9.31e0*ALOGT(I) -5.02512164e4*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,18) = EXP(9.68908955e1 -7.62e0*ALOGT(I) -3.50742017e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,19) = EXP(9.50941235e1 -7.08e0*ALOGT(I) -3.36400342e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,20) = EXP(1.38440285e2 -1.2e1*ALOGT(I) -3.00309643e3*TI(I));
    #pragma vector aligned
    for (I=1; I<=VL; I++) RKLOW(I,21) = EXP(8.93324137e1 -6.66e0*ALOGT(I) -3.52251667e3*TI(I));

}

template <class real, int MAXVL>
__declspec(target(mic)) void 
ratt_i_(int *VLp, real * RESTRICT T, real * RESTRICT RF, 
        real * RESTRICT RB, real * RESTRICT RKLOW) 
{

    const real RU=8.314510e7, RUC=RU/4.184e7, PATM=1.01325e6;
    const real SMALL = floatMin<real>(); //1.e-300;
    real PFAC, PFAC2, PFAC3,PFACI;
    real SMH[MAXVL*32], EG[MAXVL*32], EQK[MAXVL*206], EGI[MAXVL*32],
         ALOGT[MAXVL],TI[MAXVL], TI2[MAXVL],TMP[MAXVL];
    int VL = *VLp;
    int I, K, N, VL31;

    vrda_log_(VLp,T,ALOGT);

    for (I=1; I<=VL; I++) TI(I) = 1.0e0/T(I);
    for (I=1; I<=VL; I++) TI2(I) = TI(I)*TI(I);

    for (I=1; I<=VL; I++) RF(I,1) = EXP(3.20498617e1 -7.25286183e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,2) = EXP(1.08197783e1 +2.67e0*ALOGT(I) -3.16523284e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,3) = EXP(1.9190789e1 +1.51e0*ALOGT(I) -1.72603317e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,4) = EXP(1.0482906e1 +2.4e0*ALOGT(I) +1.06178717e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,5) = 1.e18*TI(I);
    for (I=1; I<=VL; I++) RF(I,6) = EXP(3.90385861e1 -6.e-1*ALOGT(I));
    for (I=1; I<=VL; I++) RF(I,7) = EXP(4.55408762e1 -1.25e0*ALOGT(I));
    for (I=1; I<=VL; I++) RF(I,8) = 5.5e20*TI2(I);
    for (I=1; I<=VL; I++) RF(I,9) = 2.2e22*TI2(I);
    for (I=1; I<=VL; I++) RF(I,10) = 5.e17*TI(I);
    for (I=1; I<=VL; I++) RF(I,11) = 1.2e17*TI(I);
    for (I=1; I<=VL; I++) RF(I,12) = EXP(4.24761511e1 -8.6e-1*ALOGT(I));
    for (I=1; I<=VL; I++) RF(I,13) = EXP(4.71503141e1 -1.72e0*ALOGT(I));
    for (I=1; I<=VL; I++) RF(I,14) = EXP(4.42511034e1 -7.6e-1*ALOGT(I));
    for (I=1; I<=VL; I++) RF(I,15) = EXP(4.47046282e1 -1.24e0*ALOGT(I));
    for (I=1; I<=VL; I++) RF(I,16) = EXP(3.19350862e1 -3.7e-1*ALOGT(I));
    for (I=1; I<=VL; I++) RF(I,17) = EXP(2.90097872e1 -3.37658384e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,18) = EXP(3.04404238e1 -4.12637667e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,19) = EXP(3.18908801e1 -1.50965e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,20) = 2.e13;
    for (I=1; I<=VL; I++) RF(I,21) = EXP(3.14683206e1 +2.51608334e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,22) = EXP(2.55908003e1 +8.20243168e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,23) = EXP(3.36712758e1 -6.03860001e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,24) = EXP(1.6308716e1 +2.e0*ALOGT(I) -2.61672667e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,25) = EXP(2.99336062e1 -1.81158e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,26) = EXP(1.60803938e1 +2.e0*ALOGT(I) -2.01286667e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,27) = EXP(2.81906369e1 -1.61029334e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,28) = EXP(3.39940492e1 -4.81075134e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,29) = EXP(3.40312786e1 -1.50965e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,30) = EXP(1.76783433e1 +1.228e0*ALOGT(I) -3.52251667e1*TI(I));
    for (I=1; I<=VL; I++) RF(I,31) = EXP(1.75767107e1 +1.5e0*ALOGT(I) -4.00560467e4*TI(I));
    for (I=1; I<=VL; I++) RF(I,32) = EXP(2.85473118e1 -2.40537567e4*TI(I));
    for (I=1; I<=VL; I++) RF(I,33) = EXP(3.26416564e1 -1.18759134e4*TI(I));
    for (I=1; I<=VL; I++) RF(I,34) = 5.7e13;
    for (I=1; I<=VL; I++) RF(I,35) = 3.e13;
    for (I=1; I<=VL; I++) RF(I,36) = EXP(1.85223344e1 +1.79e0*ALOGT(I) -8.40371835e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,37) = EXP(2.93732401e1 +3.79928584e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,38) = 3.3e13;
    for (I=1; I<=VL; I++) RF(I,39) = 5.e13;
    for (I=1; I<=VL; I++) RF(I,40) = EXP(2.88547965e1 -3.47219501e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,41) = EXP(2.77171988e1 +4.8e-1*ALOGT(I) +1.30836334e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,42) = 7.34e13;
    for (I=1; I<=VL; I++) RF(I,43) = 3.e13;
    for (I=1; I<=VL; I++) RF(I,44) = 3.e13;
    for (I=1; I<=VL; I++) RF(I,45) = 5.e13;
    for (I=1; I<=VL; I++) RF(I,46) = EXP(3.9769885e1 -1.e0*ALOGT(I) -8.55468335e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,47) = EXP(2.96591694e1 -2.01286667e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,48) = EXP(3.77576522e1 -8.e-1*ALOGT(I));
    for (I=1; I<=VL; I++) RF(I,49) = EXP(1.31223634e1 +2.e0*ALOGT(I) -3.63825651e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,50) = 8.e13;
    for (I=1; I<=VL; I++) TMP(I) = EXP(-7.54825001e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,51) = 1.056e13 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,52) = 2.64e12 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,53) = 2.e13;
    for (I=1; I<=VL; I++) RF(I,54) = EXP(1.62403133e1 +2.e0*ALOGT(I) -1.50965e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,55) = 2.e13;
    for (I=1; I<=VL; I++) RF(I,56) = EXP(2.74203001e1 +5.e-1*ALOGT(I) -2.26950717e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,57) = 4.e13;
    for (I=1; I<=VL; I++) RF(I,58) = 3.2e13;
    for (I=1; I<=VL; I++) RF(I,59) = EXP(3.03390713e1 -3.01930001e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,60) = 3.e13;
    for (I=1; I<=VL; I++) RF(I,61) = 1.5e13;
    for (I=1; I<=VL; I++) RF(I,62) = 1.5e13;
    for (I=1; I<=VL; I++) RF(I,63) = 3.e13;
    for (I=1; I<=VL; I++) RF(I,64) = 7.e13;
    for (I=1; I<=VL; I++) RF(I,65) = 2.8e13;
    for (I=1; I<=VL; I++) RF(I,66) = 1.2e13;
    for (I=1; I<=VL; I++) RF(I,67) = 3.e13;
    for (I=1; I<=VL; I++) RF(I,68) = 9.e12;
    for (I=1; I<=VL; I++) RF(I,69) = 7.e12;
    for (I=1; I<=VL; I++) RF(I,70) = 1.4e13;
    for (I=1; I<=VL; I++) RF(I,71) = EXP(2.7014835e1 +4.54e-1*ALOGT(I) -1.30836334e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,72) = EXP(2.38587601e1 +1.05e0*ALOGT(I) -1.64803459e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,73) = EXP(3.12945828e1 -1.781387e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,74) = EXP(2.19558261e1 +1.18e0*ALOGT(I) +2.2493785e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,75) = EXP(3.22361913e1 -2.01286667e4*TI(I));
    for (I=1; I<=VL; I++) TMP(I) = EXP(-4.02573334e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,76) = 1.e12 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,127) = 5.e13 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,129) = 1.e13 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,77) = EXP(3.21806786e1 +2.59156584e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,78) = EXP(3.70803784e1 -6.3e-1*ALOGT(I) -1.92731984e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,79) = 8.43e13;
    for (I=1; I<=VL; I++) RF(I,80) = EXP(1.78408622e1 +1.6e0*ALOGT(I) -2.72743434e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,81) = 2.501e13;
    for (I=1; I<=VL; I++) RF(I,82) = EXP(3.10595094e1 -1.449264e4*TI(I));
    for (I=1; I<=VL; I++) RF(I,83) = EXP(2.43067848e1 -4.49875701e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,84) = 1.e12;
    for (I=1; I<=VL; I++) RF(I,85) = 1.34e13;
    for (I=1; I<=VL; I++) RF(I,86) = EXP(1.01064284e1 +2.47e0*ALOGT(I) -2.60666234e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,87) = 3.e13;
    for (I=1; I<=VL; I++) RF(I,88) = 8.48e12;
    for (I=1; I<=VL; I++) RF(I,89) = 1.8e13;
    for (I=1; I<=VL; I++) RF(I,90) = EXP(8.10772006e0 +2.81e0*ALOGT(I) -2.94884967e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,91) = 4.e13;
    for (I=1; I<=VL; I++) TMP(I) = EXP(2.86833501e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,92) = 1.2e13 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,107) = 1.6e13 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,93) = EXP(3.75927776e1 -9.7e-1*ALOGT(I) -3.11994334e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,94) = EXP(2.9238457e1 +1.e-1*ALOGT(I) -5.33409668e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,95) = 5.e13;
    for (I=1; I<=VL; I++) RF(I,96) = 2.e13;
    for (I=1; I<=VL; I++) RF(I,97) = 3.2e13;
    for (I=1; I<=VL; I++) RF(I,98) = 1.6e13;
    for (I=1; I<=VL; I++) RF(I,99) = 1.e13;
    for (I=1; I<=VL; I++) RF(I,100) = 5.e12;
    for (I=1; I<=VL; I++) RF(I,101) = EXP(-2.84796532e1 +7.6e0*ALOGT(I) +1.77635484e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,102) = EXP(2.03077504e1 +1.62e0*ALOGT(I) -5.45486868e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,103) = EXP(2.07430685e1 +1.5e0*ALOGT(I) -4.32766334e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,104) = EXP(1.84206807e1 +1.6e0*ALOGT(I) -1.570036e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,105) = 6.e13;
    for (I=1; I<=VL; I++) RF(I,106) = EXP(1.47156719e1 +2.e0*ALOGT(I) -4.16160184e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,108) = 1.e14;
    for (I=1; I<=VL; I++) RF(I,109) = 1.e14;
    for (I=1; I<=VL; I++) RF(I,110) = EXP(2.81010247e1 -4.29747034e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,111) = 5.e13;
    for (I=1; I<=VL; I++) RF(I,112) = 3.e13;
    for (I=1; I<=VL; I++) RF(I,113) = 1.e13;
    for (I=1; I<=VL; I++) RF(I,114) = EXP(3.43156328e1 -5.2e-1*ALOGT(I) -2.55382459e4*TI(I));
    for (I=1; I<=VL; I++) RF(I,115) = EXP(1.97713479e1 +1.62e0*ALOGT(I) -1.86432818e4*TI(I));
    for (I=1; I<=VL; I++) TMP(I) = EXP(2.e0*ALOGT(I) -9.56111669e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,116) = 1.632e7 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,117) = 4.08e6 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,118) = EXP(-8.4310155e0 +4.5e0*ALOGT(I) +5.03216668e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,119) = EXP(-7.6354939e0 +4.e0*ALOGT(I) +1.00643334e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,120) = EXP(1.61180957e1 +2.e0*ALOGT(I) -3.01930001e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,121) = EXP(1.27430637e2 -1.182e1*ALOGT(I) -1.79799315e4*TI(I));
    for (I=1; I<=VL; I++) RF(I,122) = 1.e14;
    for (I=1; I<=VL; I++) RF(I,123) = 1.e14;
    for (I=1; I<=VL; I++) RF(I,124) = 2.e13;
    for (I=1; I<=VL; I++) RF(I,125) = 1.e13;
    for (I=1; I<=VL; I++) RF(I,126) = EXP(3.34301138e1 -6.e-2*ALOGT(I) -4.27734167e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,128) = EXP(2.11287309e1 +1.43e0*ALOGT(I) -1.35365284e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,130) = EXP(2.81906369e1 -6.79342501e2*TI(I));
    for (I=1; I<=VL; I++) TMP(I) = EXP(-1.00643334e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,131) = 7.5e12 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,152) = 1.e13 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,186) = 2.e13 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,132) = EXP(2.94360258e1 +2.7e-1*ALOGT(I) -1.40900667e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,133) = 3.e13;
    for (I=1; I<=VL; I++) RF(I,134) = 6.e13;
    for (I=1; I<=VL; I++) RF(I,135) = 4.8e13;
    for (I=1; I<=VL; I++) RF(I,136) = 4.8e13;
    for (I=1; I<=VL; I++) RF(I,137) = 3.011e13;
    for (I=1; I<=VL; I++) RF(I,138) = EXP(1.41081802e1 +1.61e0*ALOGT(I) +1.9293327e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,139) = EXP(2.64270483e1 +2.9e-1*ALOGT(I) -5.53538334e0*TI(I));
    for (I=1; I<=VL; I++) RF(I,140) = EXP(3.83674178e1 -1.39e0*ALOGT(I) -5.08248834e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,141) = 1.e13;
    for (I=1; I<=VL; I++) RF(I,142) = EXP(2.32164713e1 +2.99917134e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,143) = 9.033e13;
    for (I=1; I<=VL; I++) RF(I,144) = 3.92e11;
    for (I=1; I<=VL; I++) RF(I,145) = 2.5e13;
    for (I=1; I<=VL; I++) RF(I,146) = EXP(5.56675073e1 -2.83e0*ALOGT(I) -9.36888792e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,147) = EXP(9.64601125e1 -9.147e0*ALOGT(I) -2.36008617e4*TI(I));
    for (I=1; I<=VL; I++) RF(I,148) = 1.e14;
    for (I=1; I<=VL; I++) RF(I,149) = 9.e13;
    for (I=1; I<=VL; I++) TMP(I) = EXP(-2.01286667e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,150) = 2.e13 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,151) = 2.e13 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,153) = 1.4e11;
    for (I=1; I<=VL; I++) RF(I,154) = 1.8e10;
    for (I=1; I<=VL; I++) RF(I,155) = EXP(2.97104627e1 +4.4e-1*ALOGT(I) -4.46705436e4*TI(I));
    for (I=1; I<=VL; I++) RF(I,156) = EXP(2.77079822e1 +4.54e-1*ALOGT(I) -9.15854335e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,157) = EXP(1.77414365e1 +1.93e0*ALOGT(I) -6.51665585e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,158) = EXP(1.65302053e1 +1.91e0*ALOGT(I) -1.88203034e3*TI(I));
    for (I=1; I<=VL; I++) TMP(I) = EXP(1.83e0*ALOGT(I) -1.10707667e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,159) = 1.92e7 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,160) = 3.84e5 * TMP(I);
    for (I=1; I<=VL; I++) RF(I,161) = EXP(1.50964444e1 +2.e0*ALOGT(I) -1.25804167e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,162) = EXP(3.13734413e1 -3.05955734e4*TI(I));
    for (I=1; I<=VL; I++) RF(I,163) = EXP(2.83241683e1 -7.04503335e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,164) = EXP(1.61180957e1 +2.e0*ALOGT(I) -4.02573334e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,165) = EXP(3.06267534e1 -3.01930001e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,166) = 5.e13;
    for (I=1; I<=VL; I++) RF(I,167) = 5.e13;
    for (I=1; I<=VL; I++) RF(I,168) = EXP(1.23327053e1 +2.e0*ALOGT(I) -4.62959334e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,169) = EXP(2.65223585e1 -3.87476834e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,170) = EXP(4.07945264e1 -9.9e-1*ALOGT(I) -7.95082335e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,171) = 2.e12;
    for (I=1; I<=VL; I++) RF(I,172) = 1.604e13;
    for (I=1; I<=VL; I++) RF(I,173) = 8.02e13;
    for (I=1; I<=VL; I++) RF(I,174) = 2.e10;
    for (I=1; I<=VL; I++) RF(I,175) = 3.e11;
    for (I=1; I<=VL; I++) RF(I,176) = 3.e11;
    for (I=1; I<=VL; I++) RF(I,177) = 2.4e13;
    for (I=1; I<=VL; I++) RF(I,178) = EXP(2.28865889e1 -4.90133034e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,179) = 1.2e14;
    for (I=1; I<=VL; I++) RF(I,180) = EXP(1.85604427e1 +1.9e0*ALOGT(I) -3.78922151e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,181) = EXP(1.83130955e1 +1.92e0*ALOGT(I) -2.86330284e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,182) = EXP(1.50796373e1 +2.12e0*ALOGT(I) -4.37798501e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,183) = EXP(3.13199006e1 +2.76769167e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,184) = EXP(1.56303353e1 +1.74e0*ALOGT(I) -5.25861418e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,185) = 2.e14;
    for (I=1; I<=VL; I++) RF(I,187) = 2.66e12;
    for (I=1; I<=VL; I++) RF(I,188) = 6.6e12;
    for (I=1; I<=VL; I++) RF(I,189) = 6.e13;
    for (I=1; I<=VL; I++) RF(I,190) = EXP(3.02187852e1 -1.64083859e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,191) = EXP(5.11268757e1 -2.39e0*ALOGT(I) -5.62596234e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,192) = EXP(1.20435537e1 +2.5e0*ALOGT(I) -1.2530095e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,193) = EXP(1.86030023e1 +1.65e0*ALOGT(I) -1.6455185e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,194) = EXP(1.73708586e1 +1.65e0*ALOGT(I) +4.89126601e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,195) = EXP(2.59162227e1 +7.e-1*ALOGT(I) -2.95891401e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,196) = EXP(1.49469127e1 +2.e0*ALOGT(I) +1.49958567e2*TI(I));
    for (I=1; I<=VL; I++) RF(I,197) = EXP(9.16951838e0 +2.6e0*ALOGT(I) -6.99974385e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,198) = EXP(7.8845736e-1 +3.5e0*ALOGT(I) -2.85575459e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,199) = EXP(5.65703751e1 -2.92e0*ALOGT(I) -6.29272443e3*TI(I));
    for (I=1; I<=VL; I++) RF(I,200) = 1.8e12;
    for (I=1; I<=VL; I++) RF(I,201) = 9.6e13;
    for (I=1; I<=VL; I++) RF(I,202) = 2.4e13;
    for (I=1; I<=VL; I++) RF(I,203) = 9.e10;
    for (I=1; I<=VL; I++) RF(I,204) = 2.4e13;
    for (I=1; I<=VL; I++) RF(I,205) = 1.1e13;
    for (I=1; I<=VL; I++) RF(I,206) = EXP(7.50436995e1 -5.22e0*ALOGT(I) -9.93701954e3*TI(I));

    rdsmh_i_<real,MAXVL>(VLp, T, SMH);
    if(VL==MAXVL) 
    {
        VL31 = 31*MAXVL;
        vrda_exp_(&VL31,&(SMH(1,1)),&(EG(1,1)));
    }
    else 
    {
        for (N = 1; N<= 31; N++) 
        {
            for (I=0; I<(*(VLp)); I++) EG[MAXVL*(N-1)+I] = EXP(SMH[MAXVL*(N-1)+I]);
        }
    }

    for (I=1; I<=VL; I++) 
    {
        PFAC = PATM / (RU*T(I));
        PFACI = (RU*T(I))/PATM;
        PFAC2 = PFAC*PFAC;
        PFAC3 = PFAC2*PFAC;

        EQK(I,1)=EG(I,3)*EG(I,5)*EGI(I,2)*EGI(I,4);
        EQK(I,2)=EG(I,2)*EG(I,5)*EGI(I,1)*EGI(I,3);
        EQK(I,3)=EG(I,2)*EG(I,6)*EGI(I,1)*EGI(I,5);
        EQK(I,4)=EG(I,3)*EG(I,6)*EGI(I,5)*EGI(I,5);
        EQK(I,5)=EG(I,1)*EGI(I,2)*EGI(I,2)*PFACI;
        EQK(I,6)=EQK(I,5);
        EQK(I,7)=EQK(I,5);
        EQK(I,8)=EQK(I,5);
        EQK(I,9)=EG(I,6)*EGI(I,2)*EGI(I,5)*PFACI;
        EQK(I,10)=EG(I,5)*EGI(I,2)*EGI(I,3)*PFACI;
        EQK(I,11)=EG(I,4)*EGI(I,3)*EGI(I,3)*PFACI;
        EQK(I,12)=EG(I,7)*EGI(I,2)*EGI(I,4)*PFACI;
        EQK(I,13)=EQK(I,12);
        EQK(I,14)=EQK(I,12);
        EQK(I,15)=EQK(I,12);
        EQK(I,16)=EG(I,8)*EGI(I,5)*EGI(I,5)*PFACI;
        EQK(I,17)=EG(I,3)*EG(I,6)*EGI(I,2)*EGI(I,7);
        EQK(I,18)=EG(I,1)*EG(I,4)*EGI(I,2)*EGI(I,7);
        EQK(I,19)=EG(I,5)*EG(I,5)*EGI(I,2)*EGI(I,7);
        EQK(I,20)=EG(I,4)*EG(I,5)*EGI(I,3)*EGI(I,7);
        EQK(I,21)=EG(I,4)*EG(I,6)*EGI(I,5)*EGI(I,7);
        EQK(I,22)=EG(I,4)*EG(I,8)*EGI(I,7)*EGI(I,7);
        EQK(I,23)=EQK(I,22);
        EQK(I,24)=EG(I,1)*EG(I,7)*EGI(I,2)*EGI(I,8);
        EQK(I,25)=EG(I,5)*EG(I,6)*EGI(I,2)*EGI(I,8);
        EQK(I,26)=EG(I,5)*EG(I,7)*EGI(I,3)*EGI(I,8);
        EQK(I,27)=EG(I,6)*EG(I,7)*EGI(I,5)*EGI(I,8);
        EQK(I,28)=EQK(I,27);
        EQK(I,29)=EG(I,15)*EGI(I,3)*EGI(I,14)*PFACI;
        EQK(I,30)=EG(I,2)*EG(I,15)*EGI(I,5)*EGI(I,14);
        EQK(I,31)=EG(I,17)*EGI(I,1)*EGI(I,14)*PFACI;
        EQK(I,32)=EG(I,3)*EG(I,15)*EGI(I,4)*EGI(I,14);
        EQK(I,33)=EG(I,5)*EG(I,15)*EGI(I,7)*EGI(I,14);
        EQK(I,34)=EG(I,2)*EG(I,14)*EGI(I,3)*EGI(I,9);
        EQK(I,35)=EG(I,2)*EG(I,16)*EGI(I,5)*EGI(I,9);
        EQK(I,36)=EG(I,2)*EG(I,10)*EGI(I,1)*EGI(I,9);
        EQK(I,37)=EG(I,2)*EG(I,17)*EGI(I,6)*EGI(I,9);
        EQK(I,38)=EG(I,3)*EG(I,16)*EGI(I,4)*EGI(I,9);
        EQK(I,39)=EG(I,25)*EGI(I,9)*EGI(I,14)*PFACI;
        EQK(I,40)=EG(I,14)*EG(I,16)*EGI(I,9)*EGI(I,15);
        EQK(I,41)=EG(I,17)*EGI(I,2)*EGI(I,16)*PFACI;
        EQK(I,42)=EG(I,1)*EG(I,14)*EGI(I,2)*EGI(I,16);
        EQK(I,43)=EG(I,5)*EG(I,14)*EGI(I,3)*EGI(I,16);
        EQK(I,44)=EG(I,2)*EG(I,15)*EGI(I,3)*EGI(I,16);
        EQK(I,45)=EG(I,6)*EG(I,14)*EGI(I,5)*EGI(I,16);
        EQK(I,46)=EG(I,2)*EG(I,14)*EGI(I,16)*PFAC;
        EQK(I,47)=EG(I,7)*EG(I,14)*EGI(I,4)*EGI(I,16);
        EQK(I,48)=EG(I,12)*EGI(I,2)*EGI(I,10)*PFACI;
        EQK(I,49)=EG(I,2)*EG(I,12)*EGI(I,1)*EGI(I,10);
        EQK(I,50)=EG(I,2)*EG(I,16)*EGI(I,3)*EGI(I,10);
        EQK(I,51)=EG(I,5)*EG(I,16)*EGI(I,4)*EGI(I,10);
        EQK(I,52)=EG(I,2)*EG(I,2)*EG(I,15)*EGI(I,4)*EGI(I,10)*PFAC;
        EQK(I,53)=EG(I,2)*EG(I,17)*EGI(I,5)*EGI(I,10);
        EQK(I,54)=EG(I,6)*EG(I,9)*EGI(I,5)*EGI(I,10);
        EQK(I,55)=EG(I,5)*EG(I,17)*EGI(I,7)*EGI(I,10);
        EQK(I,56)=EG(I,26)*EGI(I,10)*EGI(I,14)*PFACI;
        EQK(I,57)=EG(I,2)*EG(I,19)*EGI(I,9)*EGI(I,10);
        EQK(I,58)=EG(I,1)*EG(I,19)*EGI(I,10)*EGI(I,10);
        EQK(I,59)=EG(I,10)*EGI(I,11);
        EQK(I,67)=EQK(I,59);
        EQK(I,68)=EQK(I,59);
        EQK(I,69)=EQK(I,59);
        EQK(I,60)=EG(I,1)*EG(I,9)*EGI(I,2)*EGI(I,11);
        EQK(I,61)=EG(I,1)*EG(I,14)*EGI(I,3)*EGI(I,11);
        EQK(I,62)=EG(I,2)*EG(I,16)*EGI(I,3)*EGI(I,11);
        EQK(I,63)=EG(I,2)*EG(I,17)*EGI(I,5)*EGI(I,11);
        EQK(I,64)=EG(I,2)*EG(I,12)*EGI(I,1)*EGI(I,11);
        EQK(I,65)=EG(I,2)*EG(I,5)*EG(I,14)*EGI(I,4)*EGI(I,11)*PFAC;
        EQK(I,66)=EG(I,6)*EG(I,14)*EGI(I,4)*EGI(I,11);
        EQK(I,70)=EG(I,14)*EG(I,17)*EGI(I,11)*EGI(I,15);
        EQK(I,71)=EG(I,18)*EGI(I,2)*EGI(I,17)*PFACI;
        EQK(I,72)=EG(I,1)*EG(I,16)*EGI(I,2)*EGI(I,17);
        EQK(I,73)=EG(I,5)*EG(I,16)*EGI(I,3)*EGI(I,17);
        EQK(I,74)=EG(I,6)*EG(I,16)*EGI(I,5)*EGI(I,17);
        EQK(I,75)=EG(I,7)*EG(I,16)*EGI(I,4)*EGI(I,17);
        EQK(I,76)=EG(I,8)*EG(I,16)*EGI(I,7)*EGI(I,17);
        EQK(I,77)=EG(I,2)*EG(I,26)*EGI(I,9)*EGI(I,17);
        EQK(I,78)=EG(I,13)*EGI(I,2)*EGI(I,12)*PFACI;
        EQK(I,79)=EG(I,2)*EG(I,17)*EGI(I,3)*EGI(I,12);
        EQK(I,80)=EG(I,6)*EG(I,10)*EGI(I,5)*EGI(I,12);
        EQK(I,81)=EG(I,6)*EG(I,11)*EGI(I,5)*EGI(I,12);
        EQK(I,82)=EG(I,3)*EG(I,18)*EGI(I,4)*EGI(I,12);
        EQK(I,83)=EG(I,5)*EG(I,17)*EGI(I,4)*EGI(I,12);
        EQK(I,84)=EG(I,4)*EG(I,13)*EGI(I,7)*EGI(I,12);
        EQK(I,85)=EG(I,5)*EG(I,18)*EGI(I,7)*EGI(I,12);
        EQK(I,86)=EG(I,7)*EG(I,13)*EGI(I,8)*EGI(I,12);
        EQK(I,87)=EG(I,2)*EG(I,21)*EGI(I,9)*EGI(I,12);
        EQK(I,88)=EG(I,13)*EG(I,14)*EGI(I,12)*EGI(I,16);
        EQK(I,89)=EG(I,28)*EGI(I,12)*EGI(I,16)*PFACI;
        EQK(I,90)=EG(I,13)*EG(I,16)*EGI(I,12)*EGI(I,17);
        EQK(I,91)=EG(I,2)*EG(I,22)*EGI(I,10)*EGI(I,12);
        EQK(I,92)=EG(I,2)*EG(I,22)*EGI(I,11)*EGI(I,12);
        EQK(I,93)=EG(I,24)*EGI(I,12)*EGI(I,12)*PFACI;
        EQK(I,94)=EG(I,2)*EG(I,23)*EGI(I,12)*EGI(I,12);
        EQK(I,95)=EG(I,14)*EG(I,22)*EGI(I,12)*EGI(I,25);
        EQK(I,96)=EG(I,1)*EG(I,17)*EGI(I,2)*EGI(I,18);
        EQK(I,97)=EG(I,5)*EG(I,12)*EGI(I,2)*EGI(I,18);
        EQK(I,98)=EG(I,6)*EG(I,11)*EGI(I,2)*EGI(I,18);
        EQK(I,99)=EG(I,5)*EG(I,17)*EGI(I,3)*EGI(I,18);
        EQK(I,100)=EG(I,6)*EG(I,17)*EGI(I,5)*EGI(I,18);
        EQK(I,101)=EG(I,7)*EG(I,17)*EGI(I,4)*EGI(I,18);
        EQK(I,102)=EG(I,1)*EG(I,12)*EGI(I,2)*EGI(I,13);
        EQK(I,103)=EG(I,5)*EG(I,12)*EGI(I,3)*EGI(I,13);
        EQK(I,104)=EG(I,6)*EG(I,12)*EGI(I,5)*EGI(I,13);
        EQK(I,105)=EG(I,2)*EG(I,22)*EGI(I,9)*EGI(I,13);
        EQK(I,106)=EG(I,12)*EG(I,12)*EGI(I,10)*EGI(I,13);
        EQK(I,107)=EG(I,12)*EG(I,12)*EGI(I,11)*EGI(I,13);
        EQK(I,108)=EG(I,11)*EG(I,14)*EGI(I,2)*EGI(I,25);
        EQK(I,109)=EG(I,2)*EG(I,14)*EG(I,14)*EGI(I,3)*EGI(I,25)*PFAC;
        EQK(I,110)=EG(I,5)*EG(I,14)*EG(I,14)*EGI(I,4)*EGI(I,25)*PFAC;
        EQK(I,111)=EG(I,14)*EG(I,19)*EGI(I,9)*EGI(I,25);
        EQK(I,112)=EG(I,14)*EG(I,21)*EGI(I,10)*EGI(I,25);
        EQK(I,113)=EG(I,14)*EG(I,14)*EG(I,19)*EGI(I,25)*EGI(I,25)*PFAC;
        EQK(I,114)=EG(I,20)*EGI(I,19);
        EQK(I,122)=1.0/EQK(I,114);
        EQK(I,115)=EG(I,2)*EG(I,19)*EGI(I,21)*PFAC;
        EQK(I,116)=EG(I,2)*EG(I,25)*EGI(I,3)*EGI(I,19);
        EQK(I,117)=EG(I,10)*EG(I,14)*EGI(I,3)*EGI(I,19);
        EQK(I,118)=EG(I,2)*EG(I,26)*EGI(I,5)*EGI(I,19);
        EQK(I,119)=EG(I,12)*EG(I,14)*EGI(I,5)*EGI(I,19);
        EQK(I,120)=EG(I,14)*EG(I,21)*EGI(I,16)*EGI(I,19);
        EQK(I,121)=EG(I,29)*EGI(I,12)*EGI(I,19)*PFACI;
        EQK(I,123)=EG(I,10)*EG(I,14)*EGI(I,3)*EGI(I,20);
        EQK(I,124)=EG(I,2)*EG(I,26)*EGI(I,5)*EGI(I,20);
        EQK(I,125)=EG(I,10)*EG(I,15)*EGI(I,4)*EGI(I,20);
        EQK(I,126)=EG(I,27)*EGI(I,2)*EGI(I,26)*PFACI;
        EQK(I,127)=EG(I,1)*EG(I,25)*EGI(I,2)*EGI(I,26);
        EQK(I,128)=EG(I,12)*EG(I,14)*EGI(I,2)*EGI(I,26);
        EQK(I,129)=EG(I,5)*EG(I,25)*EGI(I,3)*EGI(I,26);
        EQK(I,130)=EG(I,10)*EG(I,15)*EGI(I,3)*EGI(I,26);
        EQK(I,131)=EG(I,6)*EG(I,25)*EGI(I,5)*EGI(I,26);
        EQK(I,132)=EG(I,22)*EGI(I,2)*EGI(I,21)*PFACI;
        EQK(I,133)=EG(I,1)*EG(I,19)*EGI(I,2)*EGI(I,21);
        EQK(I,134)=EG(I,1)*EG(I,20)*EGI(I,2)*EGI(I,21);
        EQK(I,135)=EG(I,2)*EG(I,26)*EGI(I,3)*EGI(I,21);
        EQK(I,136)=EG(I,12)*EG(I,14)*EGI(I,3)*EGI(I,21);
        EQK(I,137)=EG(I,6)*EG(I,19)*EGI(I,5)*EGI(I,21);
        EQK(I,138)=EG(I,7)*EG(I,19)*EGI(I,4)*EGI(I,21);
        EQK(I,139)=EG(I,3)*EG(I,27)*EGI(I,4)*EGI(I,21);
        EQK(I,140)=EG(I,16)*EG(I,17)*EGI(I,4)*EGI(I,21);
        EQK(I,141)=EG(I,5)*EG(I,27)*EGI(I,7)*EGI(I,21);
        EQK(I,142)=EG(I,7)*EG(I,22)*EGI(I,8)*EGI(I,21);
        EQK(I,143)=EG(I,14)*EG(I,22)*EGI(I,16)*EGI(I,21);
        EQK(I,144)=EG(I,13)*EG(I,19)*EGI(I,12)*EGI(I,21);
        EQK(I,145)=EG(I,30)*EGI(I,12)*EGI(I,21)*PFACI;
        EQK(I,146)=EG(I,2)*EG(I,29)*EGI(I,12)*EGI(I,21);
        EQK(I,147)=EG(I,12)*EG(I,14)*EGI(I,27)*PFAC;
        EQK(I,148)=EG(I,28)*EGI(I,2)*EGI(I,27)*PFACI;
        EQK(I,149)=EG(I,12)*EG(I,16)*EGI(I,2)*EGI(I,27);
        EQK(I,150)=EG(I,1)*EG(I,26)*EGI(I,2)*EGI(I,27);
        EQK(I,151)=EG(I,5)*EG(I,26)*EGI(I,3)*EGI(I,27);
        EQK(I,152)=EG(I,6)*EG(I,26)*EGI(I,5)*EGI(I,27);
        EQK(I,153)=EG(I,7)*EG(I,26)*EGI(I,4)*EGI(I,27);
        EQK(I,154)=EG(I,5)*EG(I,14)*EG(I,17)*EGI(I,4)*EGI(I,27)*PFAC;
        EQK(I,155)=EG(I,1)*EG(I,20)*EGI(I,22)*PFAC;
        EQK(I,156)=EG(I,23)*EGI(I,2)*EGI(I,22)*PFACI;
        EQK(I,157)=EG(I,1)*EG(I,21)*EGI(I,2)*EGI(I,22);
        EQK(I,158)=EG(I,5)*EG(I,21)*EGI(I,3)*EGI(I,22);
        EQK(I,159)=EG(I,12)*EG(I,16)*EGI(I,3)*EGI(I,22);
        EQK(I,160)=EG(I,10)*EG(I,17)*EGI(I,3)*EGI(I,22);
        EQK(I,161)=EG(I,6)*EG(I,21)*EGI(I,5)*EGI(I,22);
        EQK(I,162)=EG(I,7)*EG(I,21)*EGI(I,4)*EGI(I,22);
        EQK(I,163)=EG(I,5)*EG(I,28)*EGI(I,7)*EGI(I,22);
        EQK(I,164)=EG(I,14)*EG(I,23)*EGI(I,16)*EGI(I,22);
        EQK(I,165)=EG(I,2)*EG(I,29)*EGI(I,10)*EGI(I,22);
        EQK(I,166)=EG(I,13)*EG(I,20)*EGI(I,11)*EGI(I,22);
        EQK(I,167)=EG(I,2)*EG(I,29)*EGI(I,11)*EGI(I,22);
        EQK(I,168)=EG(I,13)*EG(I,21)*EGI(I,12)*EGI(I,22);
        EQK(I,169)=EG(I,31)*EGI(I,12)*EGI(I,22)*PFACI;
        EQK(I,170)=EG(I,24)*EGI(I,2)*EGI(I,23)*PFACI;
        EQK(I,171)=EG(I,1)*EG(I,22)*EGI(I,2)*EGI(I,23);
        EQK(I,172)=EG(I,12)*EG(I,17)*EGI(I,3)*EGI(I,23);
        EQK(I,173)=EG(I,2)*EG(I,28)*EGI(I,3)*EGI(I,23);
        EQK(I,174)=EG(I,7)*EG(I,22)*EGI(I,4)*EGI(I,23);
        EQK(I,175)=EG(I,4)*EG(I,24)*EGI(I,7)*EGI(I,23);
        EQK(I,176)=EG(I,8)*EG(I,22)*EGI(I,7)*EGI(I,23);
        EQK(I,177)=EG(I,5)*EG(I,12)*EG(I,17)*EGI(I,7)*EGI(I,23)*PFAC;
        EQK(I,178)=EG(I,7)*EG(I,24)*EGI(I,8)*EGI(I,23);
        EQK(I,179)=EG(I,14)*EG(I,24)*EGI(I,16)*EGI(I,23);
        EQK(I,180)=EG(I,1)*EG(I,23)*EGI(I,2)*EGI(I,24);
        EQK(I,181)=EG(I,5)*EG(I,23)*EGI(I,3)*EGI(I,24);
        EQK(I,182)=EG(I,6)*EG(I,23)*EGI(I,5)*EGI(I,24);
        EQK(I,183)=EG(I,12)*EG(I,23)*EGI(I,11)*EGI(I,24);
        EQK(I,184)=EG(I,13)*EG(I,23)*EGI(I,12)*EGI(I,24);
        EQK(I,185)=EG(I,30)*EGI(I,2)*EGI(I,29)*PFACI;
        EQK(I,186)=EG(I,13)*EG(I,20)*EGI(I,2)*EGI(I,29);
        EQK(I,187)=EG(I,4)*EG(I,30)*EGI(I,7)*EGI(I,29);
        EQK(I,188)=EG(I,5)*EG(I,17)*EG(I,21)*EGI(I,7)*EGI(I,29)*PFAC;
        EQK(I,189)=EG(I,14)*EG(I,30)*EGI(I,16)*EGI(I,29);
        EQK(I,190)=EG(I,31)*EGI(I,2)*EGI(I,30)*PFACI;
        EQK(I,191)=EG(I,12)*EG(I,22)*EGI(I,2)*EGI(I,30);
        EQK(I,192)=EG(I,1)*EG(I,29)*EGI(I,2)*EGI(I,30);
        EQK(I,193)=EG(I,2)*EG(I,12)*EG(I,26)*EGI(I,3)*EGI(I,30)*PFAC;
        EQK(I,194)=EG(I,16)*EG(I,23)*EGI(I,3)*EGI(I,30);
        EQK(I,195)=EG(I,5)*EG(I,29)*EGI(I,3)*EGI(I,30);
        EQK(I,196)=EG(I,6)*EG(I,29)*EGI(I,5)*EGI(I,30);
        EQK(I,197)=EG(I,8)*EG(I,29)*EGI(I,7)*EGI(I,30);
        EQK(I,198)=EG(I,13)*EG(I,29)*EGI(I,12)*EGI(I,30);
        EQK(I,199)=EG(I,12)*EG(I,23)*EGI(I,2)*EGI(I,31);
        EQK(I,200)=EG(I,1)*EG(I,30)*EGI(I,2)*EGI(I,31);
        EQK(I,201)=EG(I,17)*EG(I,23)*EGI(I,3)*EGI(I,31);
        EQK(I,202)=EG(I,6)*EG(I,30)*EGI(I,5)*EGI(I,31);
        EQK(I,203)=EG(I,7)*EG(I,30)*EGI(I,4)*EGI(I,31);
        EQK(I,204)=EG(I,5)*EG(I,17)*EG(I,23)*EGI(I,7)*EGI(I,31)*PFAC;
        EQK(I,205)=EG(I,13)*EG(I,30)*EGI(I,12)*EGI(I,31);
        EQK(I,206)=EG(I,12)*EG(I,29)*EGI(I,21)*EGI(I,23);
    }

    for (I=1; I<= 206; I++) 
    {
        #pragma vector always
        for (K=1; K<=VL; K++) RB(K,I) = RF(K,I) / MAX(EQK(K,I),SMALL);
    }

    for (I=1; I<=VL; I++) RKLOW(I,1) = EXP(4.22794408e1 -9.e-1*ALOGT(I) +8.55468335e2*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,2) = EXP(6.37931383e1 -3.42e0*ALOGT(I) -4.24463259e4*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,3) = EXP(6.54619238e1 -3.74e0*ALOGT(I) -9.74227469e2*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,4) = EXP(5.55621468e1 -2.57e0*ALOGT(I) -7.17083751e2*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,5) = EXP(6.33329483e1 -3.14e0*ALOGT(I) -6.18956501e2*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,6) = EXP(7.69748493e1 -5.11e0*ALOGT(I) -3.57032226e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,7) = EXP(6.98660102e1 -4.8e0*ALOGT(I) -2.79788467e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,8) = EXP(7.68923562e1 -4.76e0*ALOGT(I) -1.22784867e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,9) = EXP(1.11312542e2 -9.588e0*ALOGT(I) -2.566405e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,10) = EXP(1.15700234e2 -9.67e0*ALOGT(I) -3.13000767e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,11) = EXP(3.54348644e1 -6.4e-1*ALOGT(I) -2.50098684e4*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,12) = EXP(6.3111756e1 -3.4e0*ALOGT(I) -1.80145126e4*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,13) = EXP(9.57409899e1 -7.64e0*ALOGT(I) -5.98827834e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,14) = EXP(6.9414025e1 -3.86e0*ALOGT(I) -1.67067934e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,15) = EXP(1.35001549e2 -1.194e1*ALOGT(I) -4.9163262e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,16) = EXP(9.14494773e1 -7.297e0*ALOGT(I) -2.36511834e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,17) = EXP(1.17075165e2 -9.31e0*ALOGT(I) -5.02512164e4*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,18) = EXP(9.68908955e1 -7.62e0*ALOGT(I) -3.50742017e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,19) = EXP(9.50941235e1 -7.08e0*ALOGT(I) -3.36400342e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,20) = EXP(1.38440285e2 -1.2e1*ALOGT(I) -3.00309643e3*TI(I));
    for (I=1; I<=VL; I++) RKLOW(I,21) = EXP(8.93324137e1 -6.66e0*ALOGT(I) -3.52251667e3*TI(I));
}
