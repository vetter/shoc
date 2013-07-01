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
qssa_i_VEC(real * RESTRICT RF, real * RESTRICT RB, real * RESTRICT XQ)
{

    int VL = MAXVL;
    int I;
    double DEN, A1_0, A1_2, A1_3, A1_4, A1_7, A2_0, A2_1, A2_3, A2_4,
           A2_6, A2_7, A3_0, A3_1, A3_2, A3_4, A3_5, A3_6, A3_8,
           A4_0, A4_1, A4_2, A4_3, A4_7, A4_8, A4_9, A5_0, A5_3,
           A6_0, A6_2, A6_3, A6_7, A7_1, A7_2, A7_4, A7_6, A7_9,
           A8_0, A8_3, A8_4, A8_10, A9_0, A9_4, A9_7, A10_0, A10_8,
           A7_0, A7_3, A3_7;

    #pragma vector aligned
    for (I=1; I<=VL; I++) 
    {
        RF(I,57) = 0.e0;
        RF(I,58) = 0.e0;
        RF(I,143) = 0.e0;
        RF(I,179) = 0.e0;
        RB(I,194) = 0.e0;
        RF(I,206) = 0.e0;

        DEN = +RF(I, 34) +RF(I, 35) +RF(I, 36) +RF(I, 37) +RF(I, 38) 
            +RF(I, 39) +RF(I, 40) +RF(I, 77) +RF(I, 87) +RF(I,105) 
            +RF(I,111) 
            +RB(I, 54) +RB(I, 60);
        A1_0 = ( +RB(I, 34) +RB(I, 37) +RB(I, 39) +RB(I, 57) +RB(I, 77) 
                +RB(I,105) +RB(I,111) )/DEN;
        A1_2 = ( +RB(I, 36) +RF(I, 54) )/DEN;
        A1_3 = ( +RF(I, 60) )/DEN;
        A1_4 = ( +RB(I, 35) +RB(I, 38) +RB(I, 40) )/DEN;
        A1_7 = ( +RB(I, 87) )/DEN;

        DEN = +RF(I, 48) +RF(I, 49) +RF(I, 50) +RF(I, 51) +RF(I, 52) 
            +RF(I, 53) +RF(I, 54) +RF(I, 55) +RF(I, 56) +RF(I, 91) 
            +RF(I,106) 
            +RF(I,112) +RF(I,165) +RB(I, 36) +RB(I, 59) +RB(I, 67) 
            +RB(I, 68) 
            +RB(I, 69) +RB(I, 80) +RB(I,117) +RB(I,123) +RB(I,125) 
            +RB(I,130) 
            +RB(I,160);
        A2_0 = ( +RB(I, 48) +RB(I, 49) +RB(I, 52) +RB(I, 53) +RB(I, 55) 
                +RB(I, 56) +RB(I, 57) +RB(I, 58) +RB(I, 58) +RF(I, 80) 
                +RB(I, 91) 
                +RB(I,106) +RF(I,117) +RF(I,130) +RF(I,160) +RB(I,165) )/DEN;
        A2_1 = ( +RF(I, 36) +RB(I, 54) )/DEN;
        A2_3 = ( +RF(I, 59) +RF(I, 67) +RF(I, 68) +RF(I, 69) )/DEN;
        A2_4 = ( +RB(I, 50) +RB(I, 51) )/DEN;
        A2_6 = ( +RF(I,123) +RF(I,125) )/DEN;
        A2_7 = ( +RB(I,112) )/DEN;

        DEN = +RF(I, 59) +RF(I, 60) +RF(I, 61) +RF(I, 62) +RF(I, 63) 
            +RF(I, 64) +RF(I, 65) +RF(I, 66) +RF(I, 67) +RF(I, 68)
            +RF(I, 69) 
            +RF(I, 70) +RF(I, 92) +RF(I,107) +RF(I,166) +RF(I,167)
            +RF(I,183) 
            +RB(I, 81) +RB(I, 98) +RB(I,108);
        A3_0 = ( +RB(I, 61) +RB(I, 63) +RB(I, 64) +RB(I, 65) +RB(I, 66) 
                +RB(I, 70) +RF(I, 81) +RB(I, 92) +RB(I,107) +RF(I,108) 
                +RB(I,167) )/DEN;
        A3_1 = ( +RB(I, 60) )/DEN;
        A3_2 = ( +RB(I, 59) +RB(I, 67) +RB(I, 68) +RB(I, 69) )/DEN;
        A3_4 = ( +RB(I, 62) )/DEN;
        A3_5 = ( +RF(I, 98) )/DEN;
        A3_6 = ( +RB(I,166) )/DEN;
        A3_8 = ( +RB(I,183) )/DEN;

        DEN = +RF(I, 41) +RF(I, 42) +RF(I, 43) +RF(I, 44) +RF(I, 45) 
            +RF(I, 46) +RF(I, 47) +RF(I, 88) +RF(I, 89) +RF(I,120) 
            +RF(I,164) 
            +RF(I,189) +RB(I, 35) +RB(I, 38) +RB(I, 40) +RB(I, 50) 
            +RB(I, 51) 
            +RB(I, 62) +RB(I, 72) +RB(I, 73) +RB(I, 74) +RB(I, 75) 
            +RB(I, 76) 
            +RB(I, 90) +RB(I,140) +RB(I,149) +RB(I,159);
        A4_0 = ( +RB(I, 41) +RB(I, 42) +RB(I, 43) +RB(I, 44) +RB(I, 45) 
                +RB(I, 46) +RB(I, 47) +RF(I, 72) +RF(I, 73) +RF(I, 74) 
                +RF(I, 75) 
                +RF(I, 76) +RB(I, 88) +RB(I, 89) +RF(I, 90) +RB(I,143) 
                +RF(I,159) 
                +RB(I,179) +RB(I,189) +RF(I,194) )/DEN;
        A4_1 = ( +RF(I, 35) +RF(I, 38) +RF(I, 40) )/DEN;
        A4_2 = ( +RF(I, 50) +RF(I, 51) )/DEN;
        A4_3 = ( +RF(I, 62) )/DEN;
        A4_7 = ( +RB(I,120) +RF(I,140) )/DEN;
        A4_8 = ( +RB(I,164) )/DEN;
        A4_9 = ( +RF(I,149) )/DEN;

        DEN = +RF(I, 96) +RF(I, 97) +RF(I, 98) +RF(I, 99) +RF(I,100) 
            +RF(I,101) +RB(I, 71) +RB(I, 82) +RB(I, 85);
        A5_0 = ( +RF(I, 71) +RF(I, 82) +RF(I, 85) +RB(I, 96) +RB(I, 97) 
                +RB(I, 99) +RB(I,100) +RB(I,101) )/DEN;
        A5_3 = ( +RB(I, 98) )/DEN;

        DEN = +RF(I,122) +RF(I,123) +RF(I,124) +RF(I,125) +RB(I,114) 
            +RB(I,134) +RB(I,155) +RB(I,166) +RB(I,186);
        A6_0 = ( +RF(I,114) +RB(I,122) +RB(I,124) +RF(I,155) 
                +RF(I,186) )/DEN;
        A6_2 = ( +RB(I,123) +RB(I,125) )/DEN;
        A6_3 = ( +RF(I,166) )/DEN;
        A6_7 = ( +RF(I,134) )/DEN;

        DEN = +RF(I,115) +RF(I,132) +RF(I,133) +RF(I,134) +RF(I,135) 
            +RF(I,136) +RF(I,137) +RF(I,138) +RF(I,139) +RF(I,140) 
            +RF(I,141) 
            +RF(I,142) +RF(I,144) +RF(I,145) +RF(I,146) +RB(I, 87) 
            +RB(I,112) 
            +RB(I,120) +RB(I,157) +RB(I,158) +RB(I,161) +RB(I,162) 
            +RB(I,168) 
            +RB(I,188);
        A7_0 = ( +RB(I,115) +RB(I,132) +RB(I,133) +RB(I,135) +RB(I,136) 
                +RB(I,137) +RB(I,138) +RB(I,142) +RB(I,143) +RB(I,144) 
                +RB(I,145) 
                +RB(I,146) +RF(I,157) +RF(I,158) +RF(I,161) +RF(I,162) 
                +RF(I,168) 
                +RF(I,188) +RB(I,206) )/DEN;
        A7_1 = ( +RF(I, 87) )/DEN;
        A7_2 = ( +RF(I,112) )/DEN;
        A7_4 = ( +RF(I,120) +RB(I,140) )/DEN;
        A7_6 = ( +RB(I,134) )/DEN;
        A7_9 = ( +RB(I,139) +RB(I,141) )/DEN;

        DEN = +RF(I,170) +RF(I,171) +RF(I,172) +RF(I,173) +RF(I,174) 
            +RF(I,175) +RF(I,176) +RF(I,177) +RF(I,178) +RB(I, 94) 
            +RB(I,156) 
            +RB(I,164) +RB(I,180) +RB(I,181) +RB(I,182) +RB(I,183) 
            +RB(I,184) 
            +RB(I,199) +RB(I,201) +RB(I,204);
        A8_0 = ( +RF(I, 94) +RF(I,156) +RB(I,170) +RB(I,171) +RB(I,172) 
                +RB(I,173) +RB(I,174) +RB(I,175) +RB(I,176) +RB(I,177) 
                +RB(I,178) 
                +RB(I,179) +RF(I,180) +RF(I,181) +RF(I,182) +RF(I,184) 
                +RF(I,194) 
                +RB(I,206) )/DEN;
        A8_3 = ( +RF(I,183) )/DEN;
        A8_4 = ( +RF(I,164) )/DEN;
        A8_10 = ( +RF(I,199) +RF(I,201) +RF(I,204) )/DEN;

        DEN = +RF(I,147) +RF(I,148) +RF(I,149) +RF(I,150) +RF(I,151) 
            +RF(I,152) +RF(I,153) +RF(I,154) +RB(I,126) +RB(I,139)
            +RB(I,141);
        A9_0 = ( +RF(I,126) +RB(I,147) +RB(I,148) +RB(I,150) +RB(I,151) 
                +RB(I,152) +RB(I,153) +RB(I,154) )/DEN;
        A9_4 = ( +RB(I,149) )/DEN;
        A9_7 = ( +RF(I,139) +RF(I,141) )/DEN;

        DEN = +RF(I,199) +RF(I,200) +RF(I,201) +RF(I,202) +RF(I,203) 
            +RF(I,204) +RF(I,205) +RB(I,169) +RB(I,190);
        A10_0 = ( +RF(I,169) +RF(I,190) +RB(I,200) +RB(I,202) +RB(I,203) 
                +RB(I,205) )/DEN;
        A10_8 = ( +RB(I,199) +RB(I,201) +RB(I,204) )/DEN;

        A8_0 = A8_0 + A8_10*A10_0;
        DEN = 1 -A8_10*A10_8;
        A8_0 = A8_0/DEN;
        A8_4 = A8_4/DEN;
        A8_3 = A8_3/DEN;
        A3_0 = A3_0 + A3_5*A5_0;
        DEN = 1 -A3_5*A5_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A3_2 = A3_2/DEN;
        A3_1 = A3_1/DEN;
        A3_8 = A3_8/DEN;
        A3_6 = A3_6/DEN;
        A4_0 = A4_0 + A4_9*A9_0;
        A4_7 = A4_7 + A4_9*A9_7;
        DEN = 1 -A4_9*A9_4;
        A4_0 = A4_0/DEN;
        A4_3 = A4_3/DEN;
        A4_7 = A4_7/DEN;
        A4_2 = A4_2/DEN;
        A4_1 = A4_1/DEN;
        A4_8 = A4_8/DEN;
        A7_0 = A7_0 + A7_9*A9_0;
        A7_4 = A7_4 + A7_9*A9_4;
        DEN = 1 -A7_9*A9_7;
        A7_0 = A7_0/DEN;
        A7_4 = A7_4/DEN;
        A7_2 = A7_2/DEN;
        A7_1 = A7_1/DEN;
        A7_6 = A7_6/DEN;
        A3_0 = A3_0 + A3_6*A6_0;
        A3_7 = A3_6*A6_7;
        A3_2 = A3_2 + A3_6*A6_2;
        DEN = 1 -A3_6*A6_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A3_7 = A3_7/DEN;
        A3_2 = A3_2/DEN;
        A3_1 = A3_1/DEN;
        A3_8 = A3_8/DEN;
        A7_0 = A7_0 + A7_6*A6_0;
        A7_3 = A7_6*A6_3;
        A7_2 = A7_2 + A7_6*A6_2;
        DEN = 1 -A7_6*A6_7;
        A7_0 = A7_0/DEN;
        A7_4 = A7_4/DEN;
        A7_3 = A7_3/DEN;
        A7_2 = A7_2/DEN;
        A7_1 = A7_1/DEN;
        A2_0 = A2_0 + A2_6*A6_0;
        A2_3 = A2_3 + A2_6*A6_3;
        A2_7 = A2_7 + A2_6*A6_7;
        DEN = 1 -A2_6*A6_2;
        A2_0 = A2_0/DEN;
        A2_4 = A2_4/DEN;
        A2_3 = A2_3/DEN;
        A2_7 = A2_7/DEN;
        A2_1 = A2_1/DEN;
        A4_0 = A4_0 + A4_8*A8_0;
        A4_3 = A4_3 + A4_8*A8_3;
        DEN = 1 -A4_8*A8_4;
        A4_0 = A4_0/DEN;
        A4_3 = A4_3/DEN;
        A4_7 = A4_7/DEN;
        A4_2 = A4_2/DEN;
        A4_1 = A4_1/DEN;
        A3_0 = A3_0 + A3_8*A8_0;
        A3_4 = A3_4 + A3_8*A8_4;
        DEN = 1 -A3_8*A8_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A3_7 = A3_7/DEN;
        A3_2 = A3_2/DEN;
        A3_1 = A3_1/DEN;
        A4_0 = A4_0 + A4_1*A1_0;
        A4_3 = A4_3 + A4_1*A1_3;
        A4_7 = A4_7 + A4_1*A1_7;
        A4_2 = A4_2 + A4_1*A1_2;
        DEN = 1 -A4_1*A1_4;
        A4_0 = A4_0/DEN;
        A4_3 = A4_3/DEN;
        A4_7 = A4_7/DEN;
        A4_2 = A4_2/DEN;
        A3_0 = A3_0 + A3_1*A1_0;
        A3_4 = A3_4 + A3_1*A1_4;
        A3_7 = A3_7 + A3_1*A1_7;
        A3_2 = A3_2 + A3_1*A1_2;
        DEN = 1 -A3_1*A1_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A3_7 = A3_7/DEN;
        A3_2 = A3_2/DEN;
        A7_0 = A7_0 + A7_1*A1_0;
        A7_4 = A7_4 + A7_1*A1_4;
        A7_3 = A7_3 + A7_1*A1_3;
        A7_2 = A7_2 + A7_1*A1_2;
        DEN = 1 -A7_1*A1_7;
        A7_0 = A7_0/DEN;
        A7_4 = A7_4/DEN;
        A7_3 = A7_3/DEN;
        A7_2 = A7_2/DEN;
        A2_0 = A2_0 + A2_1*A1_0;
        A2_4 = A2_4 + A2_1*A1_4;
        A2_3 = A2_3 + A2_1*A1_3;
        A2_7 = A2_7 + A2_1*A1_7;
        DEN = 1 -A2_1*A1_2;
        A2_0 = A2_0/DEN;
        A2_4 = A2_4/DEN;
        A2_3 = A2_3/DEN;
        A2_7 = A2_7/DEN;
        A4_0 = A4_0 + A4_2*A2_0;
        A4_3 = A4_3 + A4_2*A2_3;
        A4_7 = A4_7 + A4_2*A2_7;
        DEN = 1 -A4_2*A2_4;
        A4_0 = A4_0/DEN;
        A4_3 = A4_3/DEN;
        A4_7 = A4_7/DEN;
        A3_0 = A3_0 + A3_2*A2_0;
        A3_4 = A3_4 + A3_2*A2_4;
        A3_7 = A3_7 + A3_2*A2_7;
        DEN = 1 -A3_2*A2_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A3_7 = A3_7/DEN;
        A7_0 = A7_0 + A7_2*A2_0;
        A7_4 = A7_4 + A7_2*A2_4;
        A7_3 = A7_3 + A7_2*A2_3;
        DEN = 1 -A7_2*A2_7;
        A7_0 = A7_0/DEN;
        A7_4 = A7_4/DEN;
        A7_3 = A7_3/DEN;
        A4_0 = A4_0 + A4_7*A7_0;
        A4_3 = A4_3 + A4_7*A7_3;
        DEN = 1 -A4_7*A7_4;
        A4_0 = A4_0/DEN;
        A4_3 = A4_3/DEN;
        A3_0 = A3_0 + A3_7*A7_0;
        A3_4 = A3_4 + A3_7*A7_4;
        DEN = 1 -A3_7*A7_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A4_0 = A4_0 + A4_3*A3_0;
        DEN = 1 -A4_3*A3_4;
        A4_0 = A4_0/DEN;
        XQ(I,4) = A4_0;
        XQ(I,3) = A3_0 +A3_4*XQ(I,4);
        XQ(I,7) = A7_0 +A7_4*XQ(I,4) +A7_3*XQ(I,3);
        XQ(I,2) = A2_0 +A2_4*XQ(I,4) +A2_3*XQ(I,3) +A2_7*XQ(I,7);
        XQ(I,1) = A1_0 +A1_4*XQ(I,4) +A1_3*XQ(I,3) +A1_7*XQ(I,7) 
            +A1_2*XQ(I,2);
        XQ(I,8) = A8_0 +A8_4*XQ(I,4) +A8_3*XQ(I,3);
        XQ(I,6) = A6_0 +A6_3*XQ(I,3) +A6_7*XQ(I,7) +A6_2*XQ(I,2);
        XQ(I,9) = A9_0 +A9_4*XQ(I,4) +A9_7*XQ(I,7);
        XQ(I,5) = A5_0 +A5_3*XQ(I,3);
        XQ(I,10) = A10_0 +A10_8*XQ(I,8);

        RF(I, 34) = RF(I, 34)*XQ(I, 1);
        RF(I, 35) = RF(I, 35)*XQ(I, 1);
        RB(I, 35) = RB(I, 35)*XQ(I, 4);
        RF(I, 36) = RF(I, 36)*XQ(I, 1);
        RB(I, 36) = RB(I, 36)*XQ(I, 2);
        RF(I, 37) = RF(I, 37)*XQ(I, 1);
        RF(I, 38) = RF(I, 38)*XQ(I, 1);
        RB(I, 38) = RB(I, 38)*XQ(I, 4);
        RF(I, 39) = RF(I, 39)*XQ(I, 1);
        RF(I, 40) = RF(I, 40)*XQ(I, 1);
        RB(I, 40) = RB(I, 40)*XQ(I, 4);
        RF(I, 41) = RF(I, 41)*XQ(I, 4);
        RF(I, 42) = RF(I, 42)*XQ(I, 4);
        RF(I, 43) = RF(I, 43)*XQ(I, 4);
        RF(I, 44) = RF(I, 44)*XQ(I, 4);
        RF(I, 45) = RF(I, 45)*XQ(I, 4);
        RF(I, 46) = RF(I, 46)*XQ(I, 4);
        RF(I, 47) = RF(I, 47)*XQ(I, 4);
        RF(I, 48) = RF(I, 48)*XQ(I, 2);
        RF(I, 49) = RF(I, 49)*XQ(I, 2);
        RF(I, 50) = RF(I, 50)*XQ(I, 2);
        RB(I, 50) = RB(I, 50)*XQ(I, 4);
        RF(I, 51) = RF(I, 51)*XQ(I, 2);
        RB(I, 51) = RB(I, 51)*XQ(I, 4);
        RF(I, 52) = RF(I, 52)*XQ(I, 2);
        RF(I, 53) = RF(I, 53)*XQ(I, 2);
        RF(I, 54) = RF(I, 54)*XQ(I, 2);
        RB(I, 54) = RB(I, 54)*XQ(I, 1);
        RF(I, 55) = RF(I, 55)*XQ(I, 2);
        RF(I, 56) = RF(I, 56)*XQ(I, 2);
        RF(I, 59) = RF(I, 59)*XQ(I, 3);
        RB(I, 59) = RB(I, 59)*XQ(I, 2);
        RF(I, 60) = RF(I, 60)*XQ(I, 3);
        RB(I, 60) = RB(I, 60)*XQ(I, 1);
        RF(I, 61) = RF(I, 61)*XQ(I, 3);
        RF(I, 62) = RF(I, 62)*XQ(I, 3);
        RB(I, 62) = RB(I, 62)*XQ(I, 4);
        RF(I, 63) = RF(I, 63)*XQ(I, 3);
        RF(I, 64) = RF(I, 64)*XQ(I, 3);
        RF(I, 65) = RF(I, 65)*XQ(I, 3);
        RF(I, 66) = RF(I, 66)*XQ(I, 3);
        RF(I, 67) = RF(I, 67)*XQ(I, 3);
        RB(I, 67) = RB(I, 67)*XQ(I, 2);
        RF(I, 68) = RF(I, 68)*XQ(I, 3);
        RB(I, 68) = RB(I, 68)*XQ(I, 2);
        RF(I, 69) = RF(I, 69)*XQ(I, 3);
        RB(I, 69) = RB(I, 69)*XQ(I, 2);
        RF(I, 70) = RF(I, 70)*XQ(I, 3);
        RB(I, 71) = RB(I, 71)*XQ(I, 5);
        RB(I, 72) = RB(I, 72)*XQ(I, 4);
        RB(I, 73) = RB(I, 73)*XQ(I, 4);
        RB(I, 74) = RB(I, 74)*XQ(I, 4);
        RB(I, 75) = RB(I, 75)*XQ(I, 4);
        RB(I, 76) = RB(I, 76)*XQ(I, 4);
        RF(I, 77) = RF(I, 77)*XQ(I, 1);
        RB(I, 80) = RB(I, 80)*XQ(I, 2);
        RB(I, 81) = RB(I, 81)*XQ(I, 3);
        RB(I, 82) = RB(I, 82)*XQ(I, 5);
        RB(I, 85) = RB(I, 85)*XQ(I, 5);
        RF(I, 87) = RF(I, 87)*XQ(I, 1);
        RB(I, 87) = RB(I, 87)*XQ(I, 7);
        RF(I, 88) = RF(I, 88)*XQ(I, 4);
        RF(I, 89) = RF(I, 89)*XQ(I, 4);
        RB(I, 90) = RB(I, 90)*XQ(I, 4);
        RF(I, 91) = RF(I, 91)*XQ(I, 2);
        RF(I, 92) = RF(I, 92)*XQ(I, 3);
        RB(I, 94) = RB(I, 94)*XQ(I, 8);
        RF(I, 96) = RF(I, 96)*XQ(I, 5);
        RF(I, 97) = RF(I, 97)*XQ(I, 5);
        RF(I, 98) = RF(I, 98)*XQ(I, 5);
        RB(I, 98) = RB(I, 98)*XQ(I, 3);
        RF(I, 99) = RF(I, 99)*XQ(I, 5);
        RF(I,100) = RF(I,100)*XQ(I, 5);
        RF(I,101) = RF(I,101)*XQ(I, 5);
        RF(I,105) = RF(I,105)*XQ(I, 1);
        RF(I,106) = RF(I,106)*XQ(I, 2);
        RF(I,107) = RF(I,107)*XQ(I, 3);
        RB(I,108) = RB(I,108)*XQ(I, 3);
        RF(I,111) = RF(I,111)*XQ(I, 1);
        RF(I,112) = RF(I,112)*XQ(I, 2);
        RB(I,112) = RB(I,112)*XQ(I, 7);
        RB(I,114) = RB(I,114)*XQ(I, 6);
        RF(I,115) = RF(I,115)*XQ(I, 7);
        RB(I,117) = RB(I,117)*XQ(I, 2);
        RF(I,120) = RF(I,120)*XQ(I, 4);
        RB(I,120) = RB(I,120)*XQ(I, 7);
        RF(I,122) = RF(I,122)*XQ(I, 6);
        RF(I,123) = RF(I,123)*XQ(I, 6);
        RB(I,123) = RB(I,123)*XQ(I, 2);
        RF(I,124) = RF(I,124)*XQ(I, 6);
        RF(I,125) = RF(I,125)*XQ(I, 6);
        RB(I,125) = RB(I,125)*XQ(I, 2);
        RB(I,126) = RB(I,126)*XQ(I, 9);
        RB(I,130) = RB(I,130)*XQ(I, 2);
        RF(I,132) = RF(I,132)*XQ(I, 7);
        RF(I,133) = RF(I,133)*XQ(I, 7);
        RF(I,134) = RF(I,134)*XQ(I, 7);
        RB(I,134) = RB(I,134)*XQ(I, 6);
        RF(I,135) = RF(I,135)*XQ(I, 7);
        RF(I,136) = RF(I,136)*XQ(I, 7);
        RF(I,137) = RF(I,137)*XQ(I, 7);
        RF(I,138) = RF(I,138)*XQ(I, 7);
        RF(I,139) = RF(I,139)*XQ(I, 7);
        RB(I,139) = RB(I,139)*XQ(I, 9);
        RF(I,140) = RF(I,140)*XQ(I, 7);
        RB(I,140) = RB(I,140)*XQ(I, 4);
        RF(I,141) = RF(I,141)*XQ(I, 7);
        RB(I,141) = RB(I,141)*XQ(I, 9);
        RF(I,142) = RF(I,142)*XQ(I, 7);
        RF(I,144) = RF(I,144)*XQ(I, 7);
        RF(I,145) = RF(I,145)*XQ(I, 7);
        RF(I,146) = RF(I,146)*XQ(I, 7);
        RF(I,147) = RF(I,147)*XQ(I, 9);
        RF(I,148) = RF(I,148)*XQ(I, 9);
        RF(I,149) = RF(I,149)*XQ(I, 9);
        RB(I,149) = RB(I,149)*XQ(I, 4);
        RF(I,150) = RF(I,150)*XQ(I, 9);
        RF(I,151) = RF(I,151)*XQ(I, 9);
        RF(I,152) = RF(I,152)*XQ(I, 9);
        RF(I,153) = RF(I,153)*XQ(I, 9);
        RF(I,154) = RF(I,154)*XQ(I, 9);
        RB(I,155) = RB(I,155)*XQ(I, 6);
        RB(I,156) = RB(I,156)*XQ(I, 8);
        RB(I,157) = RB(I,157)*XQ(I, 7);
        RB(I,158) = RB(I,158)*XQ(I, 7);
        RB(I,159) = RB(I,159)*XQ(I, 4);
        RB(I,160) = RB(I,160)*XQ(I, 2);
        RB(I,161) = RB(I,161)*XQ(I, 7);
        RB(I,162) = RB(I,162)*XQ(I, 7);
        RF(I,164) = RF(I,164)*XQ(I, 4);
        RB(I,164) = RB(I,164)*XQ(I, 8);
        RF(I,165) = RF(I,165)*XQ(I, 2);
        RF(I,166) = RF(I,166)*XQ(I, 3);
        RB(I,166) = RB(I,166)*XQ(I, 6);
        RF(I,167) = RF(I,167)*XQ(I, 3);
        RB(I,168) = RB(I,168)*XQ(I, 7);
        RB(I,169) = RB(I,169)*XQ(I,10);
        RF(I,170) = RF(I,170)*XQ(I, 8);
        RF(I,171) = RF(I,171)*XQ(I, 8);
        RF(I,172) = RF(I,172)*XQ(I, 8);
        RF(I,173) = RF(I,173)*XQ(I, 8);
        RF(I,174) = RF(I,174)*XQ(I, 8);
        RF(I,175) = RF(I,175)*XQ(I, 8);
        RF(I,176) = RF(I,176)*XQ(I, 8);
        RF(I,177) = RF(I,177)*XQ(I, 8);
        RF(I,178) = RF(I,178)*XQ(I, 8);
        RB(I,180) = RB(I,180)*XQ(I, 8);
        RB(I,181) = RB(I,181)*XQ(I, 8);
        RB(I,182) = RB(I,182)*XQ(I, 8);
        RF(I,183) = RF(I,183)*XQ(I, 3);
        RB(I,183) = RB(I,183)*XQ(I, 8);
        RB(I,184) = RB(I,184)*XQ(I, 8);
        RB(I,186) = RB(I,186)*XQ(I, 6);
        RB(I,188) = RB(I,188)*XQ(I, 7);
        RF(I,189) = RF(I,189)*XQ(I, 4);
        RB(I,190) = RB(I,190)*XQ(I,10);
        RF(I,199) = RF(I,199)*XQ(I,10);
        RB(I,199) = RB(I,199)*XQ(I, 8);
        RF(I,200) = RF(I,200)*XQ(I,10);
        RF(I,201) = RF(I,201)*XQ(I,10);
        RB(I,201) = RB(I,201)*XQ(I, 8);
        RF(I,202) = RF(I,202)*XQ(I,10);
        RF(I,203) = RF(I,203)*XQ(I,10);
        RF(I,204) = RF(I,204)*XQ(I,10);
        RB(I,204) = RB(I,204)*XQ(I, 8);
        RF(I,205) = RF(I,205)*XQ(I,10);
    }
}

template <class real, int MAXVL>
__declspec(target(mic))      void 
qssa_i_(int *VLp, real * RESTRICT RF, real * RESTRICT RB, real * RESTRICT XQ)
{

    int VL = *VLp;
    int I;
    double DEN, A1_0, A1_2, A1_3, A1_4, A1_7, A2_0, A2_1, A2_3, A2_4,
           A2_6, A2_7, A3_0, A3_1, A3_2, A3_4, A3_5, A3_6, A3_8,
           A4_0, A4_1, A4_2, A4_3, A4_7, A4_8, A4_9, A5_0, A5_3,
           A6_0, A6_2, A6_3, A6_7, A7_1, A7_2, A7_4, A7_6, A7_9,
           A8_0, A8_3, A8_4, A8_10, A9_0, A9_4, A9_7, A10_0, A10_8,
           A7_0, A7_3, A3_7;

    for (I=1; I<=VL; I++)
    {
        RF(I,57) = 0.e0;
        RF(I,58) = 0.e0;
        RF(I,143) = 0.e0;
        RF(I,179) = 0.e0;
        RB(I,194) = 0.e0;
        RF(I,206) = 0.e0;

        DEN = +RF(I, 34) +RF(I, 35) +RF(I, 36) +RF(I, 37) +RF(I, 38) 
            +RF(I, 39) +RF(I, 40) +RF(I, 77) +RF(I, 87) +RF(I,105) 
            +RF(I,111) 
            +RB(I, 54) +RB(I, 60);
        A1_0 = ( +RB(I, 34) +RB(I, 37) +RB(I, 39) +RB(I, 57) +RB(I, 77) 
                +RB(I,105) +RB(I,111) )/DEN;
        A1_2 = ( +RB(I, 36) +RF(I, 54) )/DEN;
        A1_3 = ( +RF(I, 60) )/DEN;
        A1_4 = ( +RB(I, 35) +RB(I, 38) +RB(I, 40) )/DEN;
        A1_7 = ( +RB(I, 87) )/DEN;

        DEN = +RF(I, 48) +RF(I, 49) +RF(I, 50) +RF(I, 51) +RF(I, 52) 
            +RF(I, 53) +RF(I, 54) +RF(I, 55) +RF(I, 56) +RF(I, 91) 
            +RF(I,106) 
            +RF(I,112) +RF(I,165) +RB(I, 36) +RB(I, 59) +RB(I, 67) 
            +RB(I, 68) 
            +RB(I, 69) +RB(I, 80) +RB(I,117) +RB(I,123) +RB(I,125) 
            +RB(I,130) 
            +RB(I,160);
        A2_0 = ( +RB(I, 48) +RB(I, 49) +RB(I, 52) +RB(I, 53) +RB(I, 55) 
                +RB(I, 56) +RB(I, 57) +RB(I, 58) +RB(I, 58) +RF(I, 80) 
                +RB(I, 91) 
                +RB(I,106) +RF(I,117) +RF(I,130) +RF(I,160) +RB(I,165) )/DEN;
        A2_1 = ( +RF(I, 36) +RB(I, 54) )/DEN;
        A2_3 = ( +RF(I, 59) +RF(I, 67) +RF(I, 68) +RF(I, 69) )/DEN;
        A2_4 = ( +RB(I, 50) +RB(I, 51) )/DEN;
        A2_6 = ( +RF(I,123) +RF(I,125) )/DEN;
        A2_7 = ( +RB(I,112) )/DEN;

        DEN = +RF(I, 59) +RF(I, 60) +RF(I, 61) +RF(I, 62) +RF(I, 63) 
            +RF(I, 64) +RF(I, 65) +RF(I, 66) +RF(I, 67) +RF(I, 68)
            +RF(I, 69) 
            +RF(I, 70) +RF(I, 92) +RF(I,107) +RF(I,166) +RF(I,167)
            +RF(I,183) 
            +RB(I, 81) +RB(I, 98) +RB(I,108);
        A3_0 = ( +RB(I, 61) +RB(I, 63) +RB(I, 64) +RB(I, 65) +RB(I, 66) 
                +RB(I, 70) +RF(I, 81) +RB(I, 92) +RB(I,107) +RF(I,108) 
                +RB(I,167) )/DEN;
        A3_1 = ( +RB(I, 60) )/DEN;
        A3_2 = ( +RB(I, 59) +RB(I, 67) +RB(I, 68) +RB(I, 69) )/DEN;
        A3_4 = ( +RB(I, 62) )/DEN;
        A3_5 = ( +RF(I, 98) )/DEN;
        A3_6 = ( +RB(I,166) )/DEN;
        A3_8 = ( +RB(I,183) )/DEN;

        DEN = +RF(I, 41) +RF(I, 42) +RF(I, 43) +RF(I, 44) +RF(I, 45) 
            +RF(I, 46) +RF(I, 47) +RF(I, 88) +RF(I, 89) +RF(I,120) 
            +RF(I,164) 
            +RF(I,189) +RB(I, 35) +RB(I, 38) +RB(I, 40) +RB(I, 50) 
            +RB(I, 51) 
            +RB(I, 62) +RB(I, 72) +RB(I, 73) +RB(I, 74) +RB(I, 75) 
            +RB(I, 76) 
            +RB(I, 90) +RB(I,140) +RB(I,149) +RB(I,159);
        A4_0 = ( +RB(I, 41) +RB(I, 42) +RB(I, 43) +RB(I, 44) +RB(I, 45) 
                +RB(I, 46) +RB(I, 47) +RF(I, 72) +RF(I, 73) +RF(I, 74) 
                +RF(I, 75) 
                +RF(I, 76) +RB(I, 88) +RB(I, 89) +RF(I, 90) +RB(I,143) 
                +RF(I,159) 
                +RB(I,179) +RB(I,189) +RF(I,194) )/DEN;
        A4_1 = ( +RF(I, 35) +RF(I, 38) +RF(I, 40) )/DEN;
        A4_2 = ( +RF(I, 50) +RF(I, 51) )/DEN;
        A4_3 = ( +RF(I, 62) )/DEN;
        A4_7 = ( +RB(I,120) +RF(I,140) )/DEN;
        A4_8 = ( +RB(I,164) )/DEN;
        A4_9 = ( +RF(I,149) )/DEN;

        DEN = +RF(I, 96) +RF(I, 97) +RF(I, 98) +RF(I, 99) +RF(I,100) 
            +RF(I,101) +RB(I, 71) +RB(I, 82) +RB(I, 85);
        A5_0 = ( +RF(I, 71) +RF(I, 82) +RF(I, 85) +RB(I, 96) +RB(I, 97) 
                +RB(I, 99) +RB(I,100) +RB(I,101) )/DEN;
        A5_3 = ( +RB(I, 98) )/DEN;

        DEN = +RF(I,122) +RF(I,123) +RF(I,124) +RF(I,125) +RB(I,114) 
            +RB(I,134) +RB(I,155) +RB(I,166) +RB(I,186);
        A6_0 = ( +RF(I,114) +RB(I,122) +RB(I,124) +RF(I,155) 
                +RF(I,186) )/DEN;
        A6_2 = ( +RB(I,123) +RB(I,125) )/DEN;
        A6_3 = ( +RF(I,166) )/DEN;
        A6_7 = ( +RF(I,134) )/DEN;

        DEN = +RF(I,115) +RF(I,132) +RF(I,133) +RF(I,134) +RF(I,135) 
            +RF(I,136) +RF(I,137) +RF(I,138) +RF(I,139) +RF(I,140) 
            +RF(I,141) 
            +RF(I,142) +RF(I,144) +RF(I,145) +RF(I,146) +RB(I, 87) 
            +RB(I,112) 
            +RB(I,120) +RB(I,157) +RB(I,158) +RB(I,161) +RB(I,162) 
            +RB(I,168) 
            +RB(I,188);
        A7_0 = ( +RB(I,115) +RB(I,132) +RB(I,133) +RB(I,135) +RB(I,136) 
                +RB(I,137) +RB(I,138) +RB(I,142) +RB(I,143) +RB(I,144) 
                +RB(I,145) 
                +RB(I,146) +RF(I,157) +RF(I,158) +RF(I,161) +RF(I,162) 
                +RF(I,168) 
                +RF(I,188) +RB(I,206) )/DEN;
        A7_1 = ( +RF(I, 87) )/DEN;
        A7_2 = ( +RF(I,112) )/DEN;
        A7_4 = ( +RF(I,120) +RB(I,140) )/DEN;
        A7_6 = ( +RB(I,134) )/DEN;
        A7_9 = ( +RB(I,139) +RB(I,141) )/DEN;

        DEN = +RF(I,170) +RF(I,171) +RF(I,172) +RF(I,173) +RF(I,174) 
            +RF(I,175) +RF(I,176) +RF(I,177) +RF(I,178) +RB(I, 94) 
            +RB(I,156) 
            +RB(I,164) +RB(I,180) +RB(I,181) +RB(I,182) +RB(I,183) 
            +RB(I,184) 
            +RB(I,199) +RB(I,201) +RB(I,204);
        A8_0 = ( +RF(I, 94) +RF(I,156) +RB(I,170) +RB(I,171) +RB(I,172) 
                +RB(I,173) +RB(I,174) +RB(I,175) +RB(I,176) +RB(I,177) 
                +RB(I,178) 
                +RB(I,179) +RF(I,180) +RF(I,181) +RF(I,182) +RF(I,184) 
                +RF(I,194) 
                +RB(I,206) )/DEN;
        A8_3 = ( +RF(I,183) )/DEN;
        A8_4 = ( +RF(I,164) )/DEN;
        A8_10 = ( +RF(I,199) +RF(I,201) +RF(I,204) )/DEN;

        DEN = +RF(I,147) +RF(I,148) +RF(I,149) +RF(I,150) +RF(I,151) 
            +RF(I,152) +RF(I,153) +RF(I,154) +RB(I,126) +RB(I,139)
            +RB(I,141);
        A9_0 = ( +RF(I,126) +RB(I,147) +RB(I,148) +RB(I,150) +RB(I,151) 
                +RB(I,152) +RB(I,153) +RB(I,154) )/DEN;
        A9_4 = ( +RB(I,149) )/DEN;
        A9_7 = ( +RF(I,139) +RF(I,141) )/DEN;

        DEN = +RF(I,199) +RF(I,200) +RF(I,201) +RF(I,202) +RF(I,203) 
            +RF(I,204) +RF(I,205) +RB(I,169) +RB(I,190);
        A10_0 = ( +RF(I,169) +RF(I,190) +RB(I,200) +RB(I,202) +RB(I,203) 
                +RB(I,205) )/DEN;
        A10_8 = ( +RB(I,199) +RB(I,201) +RB(I,204) )/DEN;

        A8_0 = A8_0 + A8_10*A10_0;
        DEN = 1 -A8_10*A10_8;
        A8_0 = A8_0/DEN;
        A8_4 = A8_4/DEN;
        A8_3 = A8_3/DEN;
        A3_0 = A3_0 + A3_5*A5_0;
        DEN = 1 -A3_5*A5_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A3_2 = A3_2/DEN;
        A3_1 = A3_1/DEN;
        A3_8 = A3_8/DEN;
        A3_6 = A3_6/DEN;
        A4_0 = A4_0 + A4_9*A9_0;
        A4_7 = A4_7 + A4_9*A9_7;
        DEN = 1 -A4_9*A9_4;
        A4_0 = A4_0/DEN;
        A4_3 = A4_3/DEN;
        A4_7 = A4_7/DEN;
        A4_2 = A4_2/DEN;
        A4_1 = A4_1/DEN;
        A4_8 = A4_8/DEN;
        A7_0 = A7_0 + A7_9*A9_0;
        A7_4 = A7_4 + A7_9*A9_4;
        DEN = 1 -A7_9*A9_7;
        A7_0 = A7_0/DEN;
        A7_4 = A7_4/DEN;
        A7_2 = A7_2/DEN;
        A7_1 = A7_1/DEN;
        A7_6 = A7_6/DEN;
        A3_0 = A3_0 + A3_6*A6_0;
        A3_7 = A3_6*A6_7;
        A3_2 = A3_2 + A3_6*A6_2;
        DEN = 1 -A3_6*A6_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A3_7 = A3_7/DEN;
        A3_2 = A3_2/DEN;
        A3_1 = A3_1/DEN;
        A3_8 = A3_8/DEN;
        A7_0 = A7_0 + A7_6*A6_0;
        A7_3 = A7_6*A6_3;
        A7_2 = A7_2 + A7_6*A6_2;
        DEN = 1 -A7_6*A6_7;
        A7_0 = A7_0/DEN;
        A7_4 = A7_4/DEN;
        A7_3 = A7_3/DEN;
        A7_2 = A7_2/DEN;
        A7_1 = A7_1/DEN;
        A2_0 = A2_0 + A2_6*A6_0;
        A2_3 = A2_3 + A2_6*A6_3;
        A2_7 = A2_7 + A2_6*A6_7;
        DEN = 1 -A2_6*A6_2;
        A2_0 = A2_0/DEN;
        A2_4 = A2_4/DEN;
        A2_3 = A2_3/DEN;
        A2_7 = A2_7/DEN;
        A2_1 = A2_1/DEN;
        A4_0 = A4_0 + A4_8*A8_0;
        A4_3 = A4_3 + A4_8*A8_3;
        DEN = 1 -A4_8*A8_4;
        A4_0 = A4_0/DEN;
        A4_3 = A4_3/DEN;
        A4_7 = A4_7/DEN;
        A4_2 = A4_2/DEN;
        A4_1 = A4_1/DEN;
        A3_0 = A3_0 + A3_8*A8_0;
        A3_4 = A3_4 + A3_8*A8_4;
        DEN = 1 -A3_8*A8_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A3_7 = A3_7/DEN;
        A3_2 = A3_2/DEN;
        A3_1 = A3_1/DEN;
        A4_0 = A4_0 + A4_1*A1_0;
        A4_3 = A4_3 + A4_1*A1_3;
        A4_7 = A4_7 + A4_1*A1_7;
        A4_2 = A4_2 + A4_1*A1_2;
        DEN = 1 -A4_1*A1_4;
        A4_0 = A4_0/DEN;
        A4_3 = A4_3/DEN;
        A4_7 = A4_7/DEN;
        A4_2 = A4_2/DEN;
        A3_0 = A3_0 + A3_1*A1_0;
        A3_4 = A3_4 + A3_1*A1_4;
        A3_7 = A3_7 + A3_1*A1_7;
        A3_2 = A3_2 + A3_1*A1_2;
        DEN = 1 -A3_1*A1_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A3_7 = A3_7/DEN;
        A3_2 = A3_2/DEN;
        A7_0 = A7_0 + A7_1*A1_0;
        A7_4 = A7_4 + A7_1*A1_4;
        A7_3 = A7_3 + A7_1*A1_3;
        A7_2 = A7_2 + A7_1*A1_2;
        DEN = 1 -A7_1*A1_7;
        A7_0 = A7_0/DEN;
        A7_4 = A7_4/DEN;
        A7_3 = A7_3/DEN;
        A7_2 = A7_2/DEN;
        A2_0 = A2_0 + A2_1*A1_0;
        A2_4 = A2_4 + A2_1*A1_4;
        A2_3 = A2_3 + A2_1*A1_3;
        A2_7 = A2_7 + A2_1*A1_7;
        DEN = 1 -A2_1*A1_2;
        A2_0 = A2_0/DEN;
        A2_4 = A2_4/DEN;
        A2_3 = A2_3/DEN;
        A2_7 = A2_7/DEN;
        A4_0 = A4_0 + A4_2*A2_0;
        A4_3 = A4_3 + A4_2*A2_3;
        A4_7 = A4_7 + A4_2*A2_7;
        DEN = 1 -A4_2*A2_4;
        A4_0 = A4_0/DEN;
        A4_3 = A4_3/DEN;
        A4_7 = A4_7/DEN;
        A3_0 = A3_0 + A3_2*A2_0;
        A3_4 = A3_4 + A3_2*A2_4;
        A3_7 = A3_7 + A3_2*A2_7;
        DEN = 1 -A3_2*A2_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A3_7 = A3_7/DEN;
        A7_0 = A7_0 + A7_2*A2_0;
        A7_4 = A7_4 + A7_2*A2_4;
        A7_3 = A7_3 + A7_2*A2_3;
        DEN = 1 -A7_2*A2_7;
        A7_0 = A7_0/DEN;
        A7_4 = A7_4/DEN;
        A7_3 = A7_3/DEN;
        A4_0 = A4_0 + A4_7*A7_0;
        A4_3 = A4_3 + A4_7*A7_3;
        DEN = 1 -A4_7*A7_4;
        A4_0 = A4_0/DEN;
        A4_3 = A4_3/DEN;
        A3_0 = A3_0 + A3_7*A7_0;
        A3_4 = A3_4 + A3_7*A7_4;
        DEN = 1 -A3_7*A7_3;
        A3_0 = A3_0/DEN;
        A3_4 = A3_4/DEN;
        A4_0 = A4_0 + A4_3*A3_0;
        DEN = 1 -A4_3*A3_4;
        A4_0 = A4_0/DEN;
        XQ(I,4) = A4_0;
        XQ(I,3) = A3_0 +A3_4*XQ(I,4);
        XQ(I,7) = A7_0 +A7_4*XQ(I,4) +A7_3*XQ(I,3);
        XQ(I,2) = A2_0 +A2_4*XQ(I,4) +A2_3*XQ(I,3) +A2_7*XQ(I,7);
        XQ(I,1) = A1_0 +A1_4*XQ(I,4) +A1_3*XQ(I,3) +A1_7*XQ(I,7) 
            +A1_2*XQ(I,2);
        XQ(I,8) = A8_0 +A8_4*XQ(I,4) +A8_3*XQ(I,3);
        XQ(I,6) = A6_0 +A6_3*XQ(I,3) +A6_7*XQ(I,7) +A6_2*XQ(I,2);
        XQ(I,9) = A9_0 +A9_4*XQ(I,4) +A9_7*XQ(I,7);
        XQ(I,5) = A5_0 +A5_3*XQ(I,3);
        XQ(I,10) = A10_0 +A10_8*XQ(I,8);

        RF(I, 34) = RF(I, 34)*XQ(I, 1);
        RF(I, 35) = RF(I, 35)*XQ(I, 1);
        RB(I, 35) = RB(I, 35)*XQ(I, 4);
        RF(I, 36) = RF(I, 36)*XQ(I, 1);
        RB(I, 36) = RB(I, 36)*XQ(I, 2);
        RF(I, 37) = RF(I, 37)*XQ(I, 1);
        RF(I, 38) = RF(I, 38)*XQ(I, 1);
        RB(I, 38) = RB(I, 38)*XQ(I, 4);
        RF(I, 39) = RF(I, 39)*XQ(I, 1);
        RF(I, 40) = RF(I, 40)*XQ(I, 1);
        RB(I, 40) = RB(I, 40)*XQ(I, 4);
        RF(I, 41) = RF(I, 41)*XQ(I, 4);
        RF(I, 42) = RF(I, 42)*XQ(I, 4);
        RF(I, 43) = RF(I, 43)*XQ(I, 4);
        RF(I, 44) = RF(I, 44)*XQ(I, 4);
        RF(I, 45) = RF(I, 45)*XQ(I, 4);
        RF(I, 46) = RF(I, 46)*XQ(I, 4);
        RF(I, 47) = RF(I, 47)*XQ(I, 4);
        RF(I, 48) = RF(I, 48)*XQ(I, 2);
        RF(I, 49) = RF(I, 49)*XQ(I, 2);
        RF(I, 50) = RF(I, 50)*XQ(I, 2);
        RB(I, 50) = RB(I, 50)*XQ(I, 4);
        RF(I, 51) = RF(I, 51)*XQ(I, 2);
        RB(I, 51) = RB(I, 51)*XQ(I, 4);
        RF(I, 52) = RF(I, 52)*XQ(I, 2);
        RF(I, 53) = RF(I, 53)*XQ(I, 2);
        RF(I, 54) = RF(I, 54)*XQ(I, 2);
        RB(I, 54) = RB(I, 54)*XQ(I, 1);
        RF(I, 55) = RF(I, 55)*XQ(I, 2);
        RF(I, 56) = RF(I, 56)*XQ(I, 2);
        RF(I, 59) = RF(I, 59)*XQ(I, 3);
        RB(I, 59) = RB(I, 59)*XQ(I, 2);
        RF(I, 60) = RF(I, 60)*XQ(I, 3);
        RB(I, 60) = RB(I, 60)*XQ(I, 1);
        RF(I, 61) = RF(I, 61)*XQ(I, 3);
        RF(I, 62) = RF(I, 62)*XQ(I, 3);
        RB(I, 62) = RB(I, 62)*XQ(I, 4);
        RF(I, 63) = RF(I, 63)*XQ(I, 3);
        RF(I, 64) = RF(I, 64)*XQ(I, 3);
        RF(I, 65) = RF(I, 65)*XQ(I, 3);
        RF(I, 66) = RF(I, 66)*XQ(I, 3);
        RF(I, 67) = RF(I, 67)*XQ(I, 3);
        RB(I, 67) = RB(I, 67)*XQ(I, 2);
        RF(I, 68) = RF(I, 68)*XQ(I, 3);
        RB(I, 68) = RB(I, 68)*XQ(I, 2);
        RF(I, 69) = RF(I, 69)*XQ(I, 3);
        RB(I, 69) = RB(I, 69)*XQ(I, 2);
        RF(I, 70) = RF(I, 70)*XQ(I, 3);
        RB(I, 71) = RB(I, 71)*XQ(I, 5);
        RB(I, 72) = RB(I, 72)*XQ(I, 4);
        RB(I, 73) = RB(I, 73)*XQ(I, 4);
        RB(I, 74) = RB(I, 74)*XQ(I, 4);
        RB(I, 75) = RB(I, 75)*XQ(I, 4);
        RB(I, 76) = RB(I, 76)*XQ(I, 4);
        RF(I, 77) = RF(I, 77)*XQ(I, 1);
        RB(I, 80) = RB(I, 80)*XQ(I, 2);
        RB(I, 81) = RB(I, 81)*XQ(I, 3);
        RB(I, 82) = RB(I, 82)*XQ(I, 5);
        RB(I, 85) = RB(I, 85)*XQ(I, 5);
        RF(I, 87) = RF(I, 87)*XQ(I, 1);
        RB(I, 87) = RB(I, 87)*XQ(I, 7);
        RF(I, 88) = RF(I, 88)*XQ(I, 4);
        RF(I, 89) = RF(I, 89)*XQ(I, 4);
        RB(I, 90) = RB(I, 90)*XQ(I, 4);
        RF(I, 91) = RF(I, 91)*XQ(I, 2);
        RF(I, 92) = RF(I, 92)*XQ(I, 3);
        RB(I, 94) = RB(I, 94)*XQ(I, 8);
        RF(I, 96) = RF(I, 96)*XQ(I, 5);
        RF(I, 97) = RF(I, 97)*XQ(I, 5);
        RF(I, 98) = RF(I, 98)*XQ(I, 5);
        RB(I, 98) = RB(I, 98)*XQ(I, 3);
        RF(I, 99) = RF(I, 99)*XQ(I, 5);
        RF(I,100) = RF(I,100)*XQ(I, 5);
        RF(I,101) = RF(I,101)*XQ(I, 5);
        RF(I,105) = RF(I,105)*XQ(I, 1);
        RF(I,106) = RF(I,106)*XQ(I, 2);
        RF(I,107) = RF(I,107)*XQ(I, 3);
        RB(I,108) = RB(I,108)*XQ(I, 3);
        RF(I,111) = RF(I,111)*XQ(I, 1);
        RF(I,112) = RF(I,112)*XQ(I, 2);
        RB(I,112) = RB(I,112)*XQ(I, 7);
        RB(I,114) = RB(I,114)*XQ(I, 6);
        RF(I,115) = RF(I,115)*XQ(I, 7);
        RB(I,117) = RB(I,117)*XQ(I, 2);
        RF(I,120) = RF(I,120)*XQ(I, 4);
        RB(I,120) = RB(I,120)*XQ(I, 7);
        RF(I,122) = RF(I,122)*XQ(I, 6);
        RF(I,123) = RF(I,123)*XQ(I, 6);
        RB(I,123) = RB(I,123)*XQ(I, 2);
        RF(I,124) = RF(I,124)*XQ(I, 6);
        RF(I,125) = RF(I,125)*XQ(I, 6);
        RB(I,125) = RB(I,125)*XQ(I, 2);
        RB(I,126) = RB(I,126)*XQ(I, 9);
        RB(I,130) = RB(I,130)*XQ(I, 2);
        RF(I,132) = RF(I,132)*XQ(I, 7);
        RF(I,133) = RF(I,133)*XQ(I, 7);
        RF(I,134) = RF(I,134)*XQ(I, 7);
        RB(I,134) = RB(I,134)*XQ(I, 6);
        RF(I,135) = RF(I,135)*XQ(I, 7);
        RF(I,136) = RF(I,136)*XQ(I, 7);
        RF(I,137) = RF(I,137)*XQ(I, 7);
        RF(I,138) = RF(I,138)*XQ(I, 7);
        RF(I,139) = RF(I,139)*XQ(I, 7);
        RB(I,139) = RB(I,139)*XQ(I, 9);
        RF(I,140) = RF(I,140)*XQ(I, 7);
        RB(I,140) = RB(I,140)*XQ(I, 4);
        RF(I,141) = RF(I,141)*XQ(I, 7);
        RB(I,141) = RB(I,141)*XQ(I, 9);
        RF(I,142) = RF(I,142)*XQ(I, 7);
        RF(I,144) = RF(I,144)*XQ(I, 7);
        RF(I,145) = RF(I,145)*XQ(I, 7);
        RF(I,146) = RF(I,146)*XQ(I, 7);
        RF(I,147) = RF(I,147)*XQ(I, 9);
        RF(I,148) = RF(I,148)*XQ(I, 9);
        RF(I,149) = RF(I,149)*XQ(I, 9);
        RB(I,149) = RB(I,149)*XQ(I, 4);
        RF(I,150) = RF(I,150)*XQ(I, 9);
        RF(I,151) = RF(I,151)*XQ(I, 9);
        RF(I,152) = RF(I,152)*XQ(I, 9);
        RF(I,153) = RF(I,153)*XQ(I, 9);
        RF(I,154) = RF(I,154)*XQ(I, 9);
        RB(I,155) = RB(I,155)*XQ(I, 6);
        RB(I,156) = RB(I,156)*XQ(I, 8);
        RB(I,157) = RB(I,157)*XQ(I, 7);
        RB(I,158) = RB(I,158)*XQ(I, 7);
        RB(I,159) = RB(I,159)*XQ(I, 4);
        RB(I,160) = RB(I,160)*XQ(I, 2);
        RB(I,161) = RB(I,161)*XQ(I, 7);
        RB(I,162) = RB(I,162)*XQ(I, 7);
        RF(I,164) = RF(I,164)*XQ(I, 4);
        RB(I,164) = RB(I,164)*XQ(I, 8);
        RF(I,165) = RF(I,165)*XQ(I, 2);
        RF(I,166) = RF(I,166)*XQ(I, 3);
        RB(I,166) = RB(I,166)*XQ(I, 6);
        RF(I,167) = RF(I,167)*XQ(I, 3);
        RB(I,168) = RB(I,168)*XQ(I, 7);
        RB(I,169) = RB(I,169)*XQ(I,10);
        RF(I,170) = RF(I,170)*XQ(I, 8);
        RF(I,171) = RF(I,171)*XQ(I, 8);
        RF(I,172) = RF(I,172)*XQ(I, 8);
        RF(I,173) = RF(I,173)*XQ(I, 8);
        RF(I,174) = RF(I,174)*XQ(I, 8);
        RF(I,175) = RF(I,175)*XQ(I, 8);
        RF(I,176) = RF(I,176)*XQ(I, 8);
        RF(I,177) = RF(I,177)*XQ(I, 8);
        RF(I,178) = RF(I,178)*XQ(I, 8);
        RB(I,180) = RB(I,180)*XQ(I, 8);
        RB(I,181) = RB(I,181)*XQ(I, 8);
        RB(I,182) = RB(I,182)*XQ(I, 8);
        RF(I,183) = RF(I,183)*XQ(I, 3);
        RB(I,183) = RB(I,183)*XQ(I, 8);
        RB(I,184) = RB(I,184)*XQ(I, 8);
        RB(I,186) = RB(I,186)*XQ(I, 6);
        RB(I,188) = RB(I,188)*XQ(I, 7);
        RF(I,189) = RF(I,189)*XQ(I, 4);
        RB(I,190) = RB(I,190)*XQ(I,10);
        RF(I,199) = RF(I,199)*XQ(I,10);
        RB(I,199) = RB(I,199)*XQ(I, 8);
        RF(I,200) = RF(I,200)*XQ(I,10);
        RF(I,201) = RF(I,201)*XQ(I,10);
        RB(I,201) = RB(I,201)*XQ(I, 8);
        RF(I,202) = RF(I,202)*XQ(I,10);
        RF(I,203) = RF(I,203)*XQ(I,10);
        RF(I,204) = RF(I,204)*XQ(I,10);
        RB(I,204) = RB(I,204)*XQ(I, 8);
        RF(I,205) = RF(I,205)*XQ(I,10);
    }
}

