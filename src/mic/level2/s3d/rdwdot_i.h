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
__declspec(target(mic))  void 
rdwdot_i_VEC(real * RESTRICT RKF, real * RESTRICT RKR, real * RESTRICT WDOT)
{

    ALIGN64      real ROP[MAXVL*206];
    int VL = MAXVL;
    int I, K;

    for (I=1; I<=206; I++) 
    {
        #pragma vector aligned
        for (K=1; K<=VL; K++) 
        {
            ROP(K,I) = RKF(K,I) - RKR(K,I);
        }
    }
    #pragma vector aligned
    for (I=1; I<=VL; I++) 
    {
        ROP(I,5) = ROP(I,5) + ROP(I,6);
        ROP(I,5) = ROP(I,5) + ROP(I,7);
        ROP(I,5) = ROP(I,5) + ROP(I,8);
        ROP(I,12) = ROP(I,12) + ROP(I,13);
        ROP(I,12) = ROP(I,12) + ROP(I,14);
        ROP(I,12) = ROP(I,12) + ROP(I,15);
        ROP(I,22) = ROP(I,22) + ROP(I,23);
        ROP(I,27) = ROP(I,27) + ROP(I,28);
        ROP(I,59) = ROP(I,59) + ROP(I,67);
        ROP(I,59) = ROP(I,59) + ROP(I,68);
        ROP(I,59) = ROP(I,59) + ROP(I,69);
        ROP(I,114) = ROP(I,114) - ROP(I,122);
        WDOT(I,1) = -ROP(I,2) -ROP(I,3) +ROP(I,5) +ROP(I,18) 
            +ROP(I,24) -ROP(I,31) -ROP(I,36) +ROP(I,42) 
            -ROP(I,49) +ROP(I,58) +ROP(I,60) +ROP(I,61) 
            -ROP(I,64) +ROP(I,72) +ROP(I,96) +ROP(I,102) 
            +ROP(I,127) +ROP(I,133) +ROP(I,134) +ROP(I,150) 
            +ROP(I,155) +ROP(I,157) +ROP(I,171) +ROP(I,180) 
            +ROP(I,192) +ROP(I,200);
        WDOT(I,2) = -ROP(I,1) +ROP(I,2) +ROP(I,3) -ROP(I,5) -ROP(I,5) 
            -ROP(I,9) -ROP(I,10) -ROP(I,12) -ROP(I,17) 
            -ROP(I,18) -ROP(I,19) -ROP(I,24) -ROP(I,25) 
            +ROP(I,30) +ROP(I,34) +ROP(I,35) +ROP(I,36) 
            +ROP(I,37) -ROP(I,41) -ROP(I,42) +ROP(I,44) 
            +ROP(I,46) -ROP(I,48) +ROP(I,49) +ROP(I,50) 
            +ROP(I,52) +ROP(I,52) +ROP(I,53) +ROP(I,57) 
            -ROP(I,60) +ROP(I,62) +ROP(I,63) +ROP(I,64) 
            +ROP(I,65) -ROP(I,71) -ROP(I,72) +ROP(I,77) 
            -ROP(I,78) +ROP(I,79) +ROP(I,87) +ROP(I,91) 
            +ROP(I,92) +ROP(I,94) -ROP(I,96) -ROP(I,97) 
            -ROP(I,98) -ROP(I,102) +ROP(I,105) -ROP(I,108) 
            +ROP(I,109) +ROP(I,115) +ROP(I,116) +ROP(I,118) 
            +ROP(I,124) -ROP(I,126) -ROP(I,127) -ROP(I,128) 
            -ROP(I,132) -ROP(I,133) -ROP(I,134) +ROP(I,135) 
            +ROP(I,146) -ROP(I,148) -ROP(I,149) -ROP(I,150) 
            -ROP(I,156) -ROP(I,157) +ROP(I,165) +ROP(I,167) 
            -ROP(I,170) -ROP(I,171) +ROP(I,173) -ROP(I,180) 
            -ROP(I,185) -ROP(I,186) -ROP(I,190) -ROP(I,191) 
            -ROP(I,192) +ROP(I,193) -ROP(I,199) -ROP(I,200);
        WDOT(I,3) = +ROP(I,1) -ROP(I,2) +ROP(I,4) -ROP(I,10) 
            -ROP(I,11) -ROP(I,11) +ROP(I,17) -ROP(I,20) 
            -ROP(I,26) -ROP(I,29) +ROP(I,32) -ROP(I,34) 
            +ROP(I,38) -ROP(I,43) -ROP(I,44) -ROP(I,50) 
            -ROP(I,61) -ROP(I,62) -ROP(I,73) -ROP(I,79) 
            +ROP(I,82) -ROP(I,99) -ROP(I,103) -ROP(I,109) 
            -ROP(I,116) -ROP(I,117) -ROP(I,123) -ROP(I,129) 
            -ROP(I,130) -ROP(I,135) -ROP(I,136) +ROP(I,139) 
            -ROP(I,151) -ROP(I,158) -ROP(I,159) -ROP(I,160) 
            -ROP(I,172) -ROP(I,173) -ROP(I,181) -ROP(I,193) 
            -ROP(I,194) -ROP(I,195) -ROP(I,201);
        WDOT(I,4) = -ROP(I,1) +ROP(I,11) -ROP(I,12) +ROP(I,18) 
            +ROP(I,20) +ROP(I,21) +ROP(I,22) -ROP(I,32) 
            -ROP(I,38) -ROP(I,47) -ROP(I,51) -ROP(I,52) 
            -ROP(I,65) -ROP(I,66) -ROP(I,75) -ROP(I,82) 
            -ROP(I,83) +ROP(I,84) -ROP(I,101) -ROP(I,110) 
            -ROP(I,125) -ROP(I,138) -ROP(I,139) -ROP(I,140) 
            -ROP(I,153) -ROP(I,154) -ROP(I,162) -ROP(I,174) 
            +ROP(I,175) +ROP(I,187) -ROP(I,203);
        WDOT(I,5) = +ROP(I,1) +ROP(I,2) -ROP(I,3) -ROP(I,4) -ROP(I,4) 
            -ROP(I,9) +ROP(I,10) -ROP(I,16) -ROP(I,16) 
            +ROP(I,19) +ROP(I,19) +ROP(I,20) -ROP(I,21) 
            +ROP(I,25) +ROP(I,26) -ROP(I,27) -ROP(I,30) 
            +ROP(I,33) -ROP(I,35) +ROP(I,43) -ROP(I,45) 
            +ROP(I,51) -ROP(I,53) -ROP(I,54) +ROP(I,55) 
            -ROP(I,63) +ROP(I,65) +ROP(I,73) -ROP(I,74) 
            -ROP(I,80) -ROP(I,81) +ROP(I,83) +ROP(I,85) 
            +ROP(I,97) +ROP(I,99) -ROP(I,100) +ROP(I,103) 
            -ROP(I,104) +ROP(I,110) -ROP(I,118) -ROP(I,119) 
            -ROP(I,124) +ROP(I,129) -ROP(I,131) -ROP(I,137) 
            +ROP(I,141) +ROP(I,151) -ROP(I,152) +ROP(I,154) 
            +ROP(I,158) -ROP(I,161) +ROP(I,163) +ROP(I,177) 
            +ROP(I,181) -ROP(I,182) +ROP(I,188) +ROP(I,195) 
            -ROP(I,196) -ROP(I,202) +ROP(I,204);
        WDOT(I,6) = +ROP(I,3) +ROP(I,4) +ROP(I,9) +ROP(I,17) 
            +ROP(I,21) +ROP(I,25) +ROP(I,27) -ROP(I,37) 
            +ROP(I,45) +ROP(I,54) +ROP(I,66) +ROP(I,74) 
            +ROP(I,80) +ROP(I,81) +ROP(I,98) +ROP(I,100) 
            +ROP(I,104) +ROP(I,131) +ROP(I,137) +ROP(I,152) 
            +ROP(I,161) +ROP(I,182) +ROP(I,196) +ROP(I,202);
        WDOT(I,7) = +ROP(I,12) -ROP(I,17) -ROP(I,18) -ROP(I,19)
            -ROP(I,20) -ROP(I,21) -ROP(I,22) -ROP(I,22) 
            +ROP(I,24) +ROP(I,26) +ROP(I,27) -ROP(I,33) 
            +ROP(I,47) -ROP(I,55) +ROP(I,75) -ROP(I,76) 
            -ROP(I,84) -ROP(I,85) +ROP(I,86) +ROP(I,101) 
            +ROP(I,138) -ROP(I,141) +ROP(I,142) +ROP(I,153) 
            +ROP(I,162) -ROP(I,163) +ROP(I,174) -ROP(I,175) 
            -ROP(I,176) -ROP(I,177) +ROP(I,178) -ROP(I,187) 
            -ROP(I,188) -ROP(I,197) +ROP(I,203) -ROP(I,204);
        WDOT(I,8) = +ROP(I,16) +ROP(I,22) -ROP(I,24) -ROP(I,25)
            -ROP(I,26) -ROP(I,27) +ROP(I,76) -ROP(I,86) 
            -ROP(I,142) +ROP(I,176) -ROP(I,178) +ROP(I,197);
        WDOT(I,9) = +ROP(I,48) +ROP(I,49) +ROP(I,64) -ROP(I,78) 
            -ROP(I,79) -ROP(I,80) -ROP(I,81) -ROP(I,82) 
            -ROP(I,83) -ROP(I,84) -ROP(I,85) -ROP(I,86) 
            -ROP(I,87) -ROP(I,88) -ROP(I,89) -ROP(I,90) 
            -ROP(I,91) -ROP(I,92) -ROP(I,93) -ROP(I,93) 
            -ROP(I,94) -ROP(I,94) -ROP(I,95) +ROP(I,97) 
            +ROP(I,102) +ROP(I,103) +ROP(I,104) +ROP(I,106) 
            +ROP(I,106) 
            +ROP(I,107) +ROP(I,107) +ROP(I,119) -ROP(I,121) 
            +ROP(I,128) +ROP(I,136) -ROP(I,144) -ROP(I,145) 
            -ROP(I,146) +ROP(I,147) +ROP(I,149) +ROP(I,159) 
            -ROP(I,168) -ROP(I,169) +ROP(I,172) +ROP(I,177) 
            +ROP(I,183) -ROP(I,184) +ROP(I,191) +ROP(I,193) 
            -ROP(I,198) +ROP(I,199) -ROP(I,205) +ROP(I,206);
        WDOT(I,10) = +ROP(I,78) +ROP(I,84) +ROP(I,86) +ROP(I,88) 
            +ROP(I,90) -ROP(I,102) -ROP(I,103) -ROP(I,104) 
            -ROP(I,105) -ROP(I,106) -ROP(I,107) +ROP(I,144) 
            +ROP(I,166) +ROP(I,168) +ROP(I,184) +ROP(I,186) 
            +ROP(I,198) +ROP(I,205);
        WDOT(I,11) = -ROP(I,29) -ROP(I,30) -ROP(I,31) -ROP(I,32) 
            -ROP(I,33) +ROP(I,34) -ROP(I,39) +ROP(I,40) 
            +ROP(I,42) +ROP(I,43) +ROP(I,45) +ROP(I,46) 
            +ROP(I,47) -ROP(I,56) +ROP(I,61) +ROP(I,65) 
            +ROP(I,66) +ROP(I,70) +ROP(I,88) +ROP(I,95) 
            +ROP(I,108) +ROP(I,109) +ROP(I,109) +ROP(I,110) 
            +ROP(I,110) 
            +ROP(I,111) +ROP(I,112) +ROP(I,113) +ROP(I,113) 
            +ROP(I,117) +ROP(I,119) +ROP(I,120) +ROP(I,123) 
            +ROP(I,128) +ROP(I,136) +ROP(I,143) +ROP(I,147) 
            +ROP(I,154) +ROP(I,164) +ROP(I,179) +ROP(I,189);
        WDOT(I,12) = +ROP(I,29) +ROP(I,30) +ROP(I,32) +ROP(I,33) 
            -ROP(I,40) +ROP(I,44) +ROP(I,52) -ROP(I,70) 
            +ROP(I,125) +ROP(I,130);
        WDOT(I,13) = +ROP(I,31) +ROP(I,37) +ROP(I,41) +ROP(I,53) 
            +ROP(I,55) +ROP(I,63) +ROP(I,70) -ROP(I,71) 
            -ROP(I,72) -ROP(I,73) -ROP(I,74) -ROP(I,75) 
            -ROP(I,76) -ROP(I,77) +ROP(I,79) +ROP(I,83) 
            -ROP(I,90) +ROP(I,96) +ROP(I,99) +ROP(I,100) 
            +ROP(I,101) +ROP(I,140) +ROP(I,154) +ROP(I,160) 
            +ROP(I,172) +ROP(I,177) +ROP(I,188) +ROP(I,201) 
            +ROP(I,204);
        WDOT(I,14) = +ROP(I,57) +ROP(I,58) +ROP(I,111) +ROP(I,113) 
            -ROP(I,114) +ROP(I,115) -ROP(I,116) -ROP(I,117) 
            -ROP(I,118) -ROP(I,119) -ROP(I,120) -ROP(I,121) 
            +ROP(I,133) +ROP(I,137) +ROP(I,138) +ROP(I,144);
        WDOT(I,15) = +ROP(I,91) +ROP(I,92) +ROP(I,95) +ROP(I,105) 
            +ROP(I,132) +ROP(I,142) +ROP(I,143) -ROP(I,155) 
            -ROP(I,156) -ROP(I,157) -ROP(I,158) -ROP(I,159) 
            -ROP(I,160) -ROP(I,161) -ROP(I,162) -ROP(I,163) 
            -ROP(I,164) -ROP(I,165) -ROP(I,166) -ROP(I,167) 
            -ROP(I,168) -ROP(I,169) +ROP(I,171) +ROP(I,174) 
            +ROP(I,176) +ROP(I,191);
        WDOT(I,16) = +ROP(I,93) +ROP(I,170) +ROP(I,175) +ROP(I,178) 
            +ROP(I,179) -ROP(I,180) -ROP(I,181) -ROP(I,182) 
            -ROP(I,183) -ROP(I,184);
        WDOT(I,17) = +ROP(I,39) -ROP(I,95) -ROP(I,108) -ROP(I,109) 
            -ROP(I,110) -ROP(I,111) -ROP(I,112) -ROP(I,113) 
            -ROP(I,113)
            +ROP(I,116) +ROP(I,127) +ROP(I,129) +ROP(I,131) ;
        WDOT(I,18) = +ROP(I,56) +ROP(I,77) +ROP(I,118) +ROP(I,124) 
            -ROP(I,126) -ROP(I,127) -ROP(I,128) -ROP(I,129) 
            -ROP(I,130) -ROP(I,131) +ROP(I,135) +ROP(I,150) 
            +ROP(I,151) +ROP(I,152) +ROP(I,153) +ROP(I,193);
        WDOT(I,19) = +ROP(I,89) +ROP(I,148) +ROP(I,163) +ROP(I,173);
        WDOT(I,20) = +ROP(I,121) +ROP(I,146) +ROP(I,165) +ROP(I,167) 
            -ROP(I,185) -ROP(I,186) -ROP(I,187) -ROP(I,188) 
            -ROP(I,189) +ROP(I,192) +ROP(I,195) +ROP(I,196) 
            +ROP(I,197) +ROP(I,198) +ROP(I,206);
        WDOT(I,21) = +ROP(I,145) +ROP(I,185) +ROP(I,187) +ROP(I,189) 
            -ROP(I,190) -ROP(I,191) -ROP(I,192) -ROP(I,193) 
            -ROP(I,194) -ROP(I,195) -ROP(I,196) -ROP(I,197) 
            -ROP(I,198) +ROP(I,200) +ROP(I,202) +ROP(I,203) 
            +ROP(I,205);
        WDOT(I,22) = 0.0;
    }
}

template <class real, int MAXVL>
__declspec(target(mic)) void 
rdwdot_i_(int *VLp, real * RESTRICT RKF, real * RESTRICT RKR, 
        real * RESTRICT WDOT)
{

    ALIGN64      real ROP[MAXVL*206];
    int VL = *VLp;
    int I, K;

    for (I=1; I<=206; I++) 
    {
        for (K=1; K<=VL; K++) 
        {
            ROP(K,I) = RKF(K,I) - RKR(K,I);
        }
    }

    for (I=1; I<=VL; I++) 
    {
        ROP(I,5) = ROP(I,5) + ROP(I,6);
        ROP(I,5) = ROP(I,5) + ROP(I,7);
        ROP(I,5) = ROP(I,5) + ROP(I,8);
        ROP(I,12) = ROP(I,12) + ROP(I,13);
        ROP(I,12) = ROP(I,12) + ROP(I,14);
        ROP(I,12) = ROP(I,12) + ROP(I,15);
        ROP(I,22) = ROP(I,22) + ROP(I,23);
        ROP(I,27) = ROP(I,27) + ROP(I,28);
        ROP(I,59) = ROP(I,59) + ROP(I,67);
        ROP(I,59) = ROP(I,59) + ROP(I,68);
        ROP(I,59) = ROP(I,59) + ROP(I,69);
        ROP(I,114) = ROP(I,114) - ROP(I,122);

        WDOT(I,1) = -ROP(I,2) -ROP(I,3) +ROP(I,5) +ROP(I,18) 
            +ROP(I,24) -ROP(I,31) -ROP(I,36) +ROP(I,42) 
            -ROP(I,49) +ROP(I,58) +ROP(I,60) +ROP(I,61) 
            -ROP(I,64) +ROP(I,72) +ROP(I,96) +ROP(I,102) 
            +ROP(I,127) +ROP(I,133) +ROP(I,134) +ROP(I,150) 
            +ROP(I,155) +ROP(I,157) +ROP(I,171) +ROP(I,180) 
            +ROP(I,192) +ROP(I,200);
        WDOT(I,2) = -ROP(I,1) +ROP(I,2) +ROP(I,3) -ROP(I,5) -ROP(I,5) 
            -ROP(I,9) -ROP(I,10) -ROP(I,12) -ROP(I,17) 
            -ROP(I,18) -ROP(I,19) -ROP(I,24) -ROP(I,25) 
            +ROP(I,30) +ROP(I,34) +ROP(I,35) +ROP(I,36) 
            +ROP(I,37) -ROP(I,41) -ROP(I,42) +ROP(I,44) 
            +ROP(I,46) -ROP(I,48) +ROP(I,49) +ROP(I,50) 
            +ROP(I,52) +ROP(I,52) +ROP(I,53) +ROP(I,57) 
            -ROP(I,60) +ROP(I,62) +ROP(I,63) +ROP(I,64) 
            +ROP(I,65) -ROP(I,71) -ROP(I,72) +ROP(I,77) 
            -ROP(I,78) +ROP(I,79) +ROP(I,87) +ROP(I,91) 
            +ROP(I,92) +ROP(I,94) -ROP(I,96) -ROP(I,97) 
            -ROP(I,98) -ROP(I,102) +ROP(I,105) -ROP(I,108) 
            +ROP(I,109) +ROP(I,115) +ROP(I,116) +ROP(I,118) 
            +ROP(I,124) -ROP(I,126) -ROP(I,127) -ROP(I,128) 
            -ROP(I,132) -ROP(I,133) -ROP(I,134) +ROP(I,135) 
            +ROP(I,146) -ROP(I,148) -ROP(I,149) -ROP(I,150) 
            -ROP(I,156) -ROP(I,157) +ROP(I,165) +ROP(I,167) 
            -ROP(I,170) -ROP(I,171) +ROP(I,173) -ROP(I,180) 
            -ROP(I,185) -ROP(I,186) -ROP(I,190) -ROP(I,191) 
            -ROP(I,192) +ROP(I,193) -ROP(I,199) -ROP(I,200);
        WDOT(I,3) = +ROP(I,1) -ROP(I,2) +ROP(I,4) -ROP(I,10) 
            -ROP(I,11) -ROP(I,11) +ROP(I,17) -ROP(I,20) 
            -ROP(I,26) -ROP(I,29) +ROP(I,32) -ROP(I,34) 
            +ROP(I,38) -ROP(I,43) -ROP(I,44) -ROP(I,50) 
            -ROP(I,61) -ROP(I,62) -ROP(I,73) -ROP(I,79) 
            +ROP(I,82) -ROP(I,99) -ROP(I,103) -ROP(I,109) 
            -ROP(I,116) -ROP(I,117) -ROP(I,123) -ROP(I,129) 
            -ROP(I,130) -ROP(I,135) -ROP(I,136) +ROP(I,139) 
            -ROP(I,151) -ROP(I,158) -ROP(I,159) -ROP(I,160) 
            -ROP(I,172) -ROP(I,173) -ROP(I,181) -ROP(I,193) 
            -ROP(I,194) -ROP(I,195) -ROP(I,201);
        WDOT(I,4) = -ROP(I,1) +ROP(I,11) -ROP(I,12) +ROP(I,18) 
            +ROP(I,20) +ROP(I,21) +ROP(I,22) -ROP(I,32) 
            -ROP(I,38) -ROP(I,47) -ROP(I,51) -ROP(I,52) 
            -ROP(I,65) -ROP(I,66) -ROP(I,75) -ROP(I,82) 
            -ROP(I,83) +ROP(I,84) -ROP(I,101) -ROP(I,110) 
            -ROP(I,125) -ROP(I,138) -ROP(I,139) -ROP(I,140) 
            -ROP(I,153) -ROP(I,154) -ROP(I,162) -ROP(I,174) 
            +ROP(I,175) +ROP(I,187) -ROP(I,203);
        WDOT(I,5) = +ROP(I,1) +ROP(I,2) -ROP(I,3) -ROP(I,4) -ROP(I,4) 
            -ROP(I,9) +ROP(I,10) -ROP(I,16) -ROP(I,16) 
            +ROP(I,19) +ROP(I,19) +ROP(I,20) -ROP(I,21) 
            +ROP(I,25) +ROP(I,26) -ROP(I,27) -ROP(I,30) 
            +ROP(I,33) -ROP(I,35) +ROP(I,43) -ROP(I,45) 
            +ROP(I,51) -ROP(I,53) -ROP(I,54) +ROP(I,55) 
            -ROP(I,63) +ROP(I,65) +ROP(I,73) -ROP(I,74) 
            -ROP(I,80) -ROP(I,81) +ROP(I,83) +ROP(I,85) 
            +ROP(I,97) +ROP(I,99) -ROP(I,100) +ROP(I,103) 
            -ROP(I,104) +ROP(I,110) -ROP(I,118) -ROP(I,119) 
            -ROP(I,124) +ROP(I,129) -ROP(I,131) -ROP(I,137) 
            +ROP(I,141) +ROP(I,151) -ROP(I,152) +ROP(I,154) 
            +ROP(I,158) -ROP(I,161) +ROP(I,163) +ROP(I,177) 
            +ROP(I,181) -ROP(I,182) +ROP(I,188) +ROP(I,195) 
            -ROP(I,196) -ROP(I,202) +ROP(I,204);
        WDOT(I,6) = +ROP(I,3) +ROP(I,4) +ROP(I,9) +ROP(I,17) 
            +ROP(I,21) +ROP(I,25) +ROP(I,27) -ROP(I,37) 
            +ROP(I,45) +ROP(I,54) +ROP(I,66) +ROP(I,74) 
            +ROP(I,80) +ROP(I,81) +ROP(I,98) +ROP(I,100) 
            +ROP(I,104) +ROP(I,131) +ROP(I,137) +ROP(I,152) 
            +ROP(I,161) +ROP(I,182) +ROP(I,196) +ROP(I,202);
        WDOT(I,7) = +ROP(I,12) -ROP(I,17) -ROP(I,18) -ROP(I,19)
            -ROP(I,20) -ROP(I,21) -ROP(I,22) -ROP(I,22) 
            +ROP(I,24) +ROP(I,26) +ROP(I,27) -ROP(I,33) 
            +ROP(I,47) -ROP(I,55) +ROP(I,75) -ROP(I,76) 
            -ROP(I,84) -ROP(I,85) +ROP(I,86) +ROP(I,101) 
            +ROP(I,138) -ROP(I,141) +ROP(I,142) +ROP(I,153) 
            +ROP(I,162) -ROP(I,163) +ROP(I,174) -ROP(I,175) 
            -ROP(I,176) -ROP(I,177) +ROP(I,178) -ROP(I,187) 
            -ROP(I,188) -ROP(I,197) +ROP(I,203) -ROP(I,204);
        WDOT(I,8) = +ROP(I,16) +ROP(I,22) -ROP(I,24) -ROP(I,25)
            -ROP(I,26) -ROP(I,27) +ROP(I,76) -ROP(I,86) 
            -ROP(I,142) +ROP(I,176) -ROP(I,178) +ROP(I,197);
        WDOT(I,9) = +ROP(I,48) +ROP(I,49) +ROP(I,64) -ROP(I,78) 
            -ROP(I,79) -ROP(I,80) -ROP(I,81) -ROP(I,82) 
            -ROP(I,83) -ROP(I,84) -ROP(I,85) -ROP(I,86) 
            -ROP(I,87) -ROP(I,88) -ROP(I,89) -ROP(I,90) 
            -ROP(I,91) -ROP(I,92) -ROP(I,93) -ROP(I,93) 
            -ROP(I,94) -ROP(I,94) -ROP(I,95) +ROP(I,97) 
            +ROP(I,102) +ROP(I,103) +ROP(I,104) +ROP(I,106) 
            +ROP(I,106) 
            +ROP(I,107) +ROP(I,107) +ROP(I,119) -ROP(I,121) 
            +ROP(I,128) +ROP(I,136) -ROP(I,144) -ROP(I,145) 
            -ROP(I,146) +ROP(I,147) +ROP(I,149) +ROP(I,159) 
            -ROP(I,168) -ROP(I,169) +ROP(I,172) +ROP(I,177) 
            +ROP(I,183) -ROP(I,184) +ROP(I,191) +ROP(I,193) 
            -ROP(I,198) +ROP(I,199) -ROP(I,205) +ROP(I,206);
        WDOT(I,10) = +ROP(I,78) +ROP(I,84) +ROP(I,86) +ROP(I,88) 
            +ROP(I,90) -ROP(I,102) -ROP(I,103) -ROP(I,104) 
            -ROP(I,105) -ROP(I,106) -ROP(I,107) +ROP(I,144) 
            +ROP(I,166) +ROP(I,168) +ROP(I,184) +ROP(I,186) 
            +ROP(I,198) +ROP(I,205);
        WDOT(I,11) = -ROP(I,29) -ROP(I,30) -ROP(I,31) -ROP(I,32) 
            -ROP(I,33) +ROP(I,34) -ROP(I,39) +ROP(I,40) 
            +ROP(I,42) +ROP(I,43) +ROP(I,45) +ROP(I,46) 
            +ROP(I,47) -ROP(I,56) +ROP(I,61) +ROP(I,65) 
            +ROP(I,66) +ROP(I,70) +ROP(I,88) +ROP(I,95) 
            +ROP(I,108) +ROP(I,109) +ROP(I,109) +ROP(I,110) 
            +ROP(I,110) 
            +ROP(I,111) +ROP(I,112) +ROP(I,113) +ROP(I,113) 
            +ROP(I,117) +ROP(I,119) +ROP(I,120) +ROP(I,123) 
            +ROP(I,128) +ROP(I,136) +ROP(I,143) +ROP(I,147) 
            +ROP(I,154) +ROP(I,164) +ROP(I,179) +ROP(I,189);
        WDOT(I,12) = +ROP(I,29) +ROP(I,30) +ROP(I,32) +ROP(I,33) 
            -ROP(I,40) +ROP(I,44) +ROP(I,52) -ROP(I,70) 
            +ROP(I,125) +ROP(I,130);
        WDOT(I,13) = +ROP(I,31) +ROP(I,37) +ROP(I,41) +ROP(I,53) 
            +ROP(I,55) +ROP(I,63) +ROP(I,70) -ROP(I,71) 
            -ROP(I,72) -ROP(I,73) -ROP(I,74) -ROP(I,75) 
            -ROP(I,76) -ROP(I,77) +ROP(I,79) +ROP(I,83) 
            -ROP(I,90) +ROP(I,96) +ROP(I,99) +ROP(I,100) 
            +ROP(I,101) +ROP(I,140) +ROP(I,154) +ROP(I,160) 
            +ROP(I,172) +ROP(I,177) +ROP(I,188) +ROP(I,201) 
            +ROP(I,204);
        WDOT(I,14) = +ROP(I,57) +ROP(I,58) +ROP(I,111) +ROP(I,113) 
            -ROP(I,114) +ROP(I,115) -ROP(I,116) -ROP(I,117) 
            -ROP(I,118) -ROP(I,119) -ROP(I,120) -ROP(I,121) 
            +ROP(I,133) +ROP(I,137) +ROP(I,138) +ROP(I,144);
        WDOT(I,15) = +ROP(I,91) +ROP(I,92) +ROP(I,95) +ROP(I,105) 
            +ROP(I,132) +ROP(I,142) +ROP(I,143) -ROP(I,155) 
            -ROP(I,156) -ROP(I,157) -ROP(I,158) -ROP(I,159) 
            -ROP(I,160) -ROP(I,161) -ROP(I,162) -ROP(I,163) 
            -ROP(I,164) -ROP(I,165) -ROP(I,166) -ROP(I,167) 
            -ROP(I,168) -ROP(I,169) +ROP(I,171) +ROP(I,174) 
            +ROP(I,176) +ROP(I,191);
        WDOT(I,16) = +ROP(I,93) +ROP(I,170) +ROP(I,175) +ROP(I,178) 
            +ROP(I,179) -ROP(I,180) -ROP(I,181) -ROP(I,182) 
            -ROP(I,183) -ROP(I,184);
        WDOT(I,17) = +ROP(I,39) -ROP(I,95) -ROP(I,108) -ROP(I,109) 
            -ROP(I,110) -ROP(I,111) -ROP(I,112) -ROP(I,113) 
            -ROP(I,113)
            +ROP(I,116) +ROP(I,127) +ROP(I,129) +ROP(I,131) ;
        WDOT(I,18) = +ROP(I,56) +ROP(I,77) +ROP(I,118) +ROP(I,124) 
            -ROP(I,126) -ROP(I,127) -ROP(I,128) -ROP(I,129) 
            -ROP(I,130) -ROP(I,131) +ROP(I,135) +ROP(I,150) 
            +ROP(I,151) +ROP(I,152) +ROP(I,153) +ROP(I,193);
        WDOT(I,19) = +ROP(I,89) +ROP(I,148) +ROP(I,163) +ROP(I,173);
        WDOT(I,20) = +ROP(I,121) +ROP(I,146) +ROP(I,165) +ROP(I,167) 
            -ROP(I,185) -ROP(I,186) -ROP(I,187) -ROP(I,188) 
            -ROP(I,189) +ROP(I,192) +ROP(I,195) +ROP(I,196) 
            +ROP(I,197) +ROP(I,198) +ROP(I,206);
        WDOT(I,21) = +ROP(I,145) +ROP(I,185) +ROP(I,187) +ROP(I,189) 
            -ROP(I,190) -ROP(I,191) -ROP(I,192) -ROP(I,193) 
            -ROP(I,194) -ROP(I,195) -ROP(I,196) -ROP(I,197) 
            -ROP(I,198) +ROP(I,200) +ROP(I,202) +ROP(I,203) 
            +ROP(I,205);
        WDOT(I,22) = 0.0;
    }
}

