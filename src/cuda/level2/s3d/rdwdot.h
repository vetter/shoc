#ifndef RDWDOT_H
#define RDWDOT_H

#include "S3D.h"

#define ROP2(a)  (RKF(a) - RKR (a))

// Contains GPU kernels for the rdwdot function, split up to reduce
// register pressure
template <class real>
__global__ void
LAUNCH_BOUNDS (RDWDOT_THRD, RDWDOT_BLK)
rdwdot_kernel (const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt)
{
    real ROP5 = ROP2(5) + ROP2(6) + ROP2(7) + ROP2(8);
    real ROP12 = ROP2(12) + ROP2(13) + ROP2(14) + ROP2(15);

    WDOT(2) = (-ROP2(1) +ROP2(2) +ROP2(3) -ROP5 -ROP5
            -ROP2(9) -ROP2(10) -ROP12 -ROP2(17)
            -ROP2(18) -ROP2(19) -ROP2(24) -ROP2(25)
            +ROP2(30) +ROP2(34) +ROP2(35) +ROP2(36)
            +ROP2(37) -ROP2(41) -ROP2(42) +ROP2(44)
            +ROP2(46) -ROP2(48) +ROP2(49) +ROP2(50)
            +ROP2(52) +ROP2(52) +ROP2(53) +ROP2(57)
            -ROP2(60) +ROP2(62) +ROP2(63) +ROP2(64)
            +ROP2(65) -ROP2(71) -ROP2(72) +ROP2(77)
            -ROP2(78) +ROP2(79) +ROP2(87) +ROP2(91)
            +ROP2(92) +ROP2(94) -ROP2(96) -ROP2(97)
            -ROP2(98) -ROP2(102) +ROP2(105) -ROP2(108)
            +ROP2(109) +ROP2(115) +ROP2(116) +ROP2(118)
            +ROP2(124) -ROP2(126) -ROP2(127) -ROP2(128)
            -ROP2(132) -ROP2(133) -ROP2(134) +ROP2(135)
            +ROP2(146) -ROP2(148) -ROP2(149) -ROP2(150)
            -ROP2(156) -ROP2(157) +ROP2(165) +ROP2(167)
            -ROP2(170) -ROP2(171) +ROP2(173) -ROP2(180)
            -ROP2(185) -ROP2(186) -ROP2(190) -ROP2(191)
            -ROP2(192) +ROP2(193) -ROP2(199) -ROP2(200))*rateconv *molwt[1];
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RDWDOT2_THRD, RDWDOT2_BLK)
rdwdot2_kernel (const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt)
{

    WDOT(21) = (ROP2(145) +ROP2(185) +ROP2(187) +ROP2(189)
            -ROP2(190) -ROP2(191) -ROP2(192) -ROP2(193)
            -ROP2(194) -ROP2(195) -ROP2(196) -ROP2(197)
            -ROP2(198) +ROP2(200) +ROP2(202) +ROP2(203)
            +ROP2(205))*rateconv *molwt[20];

    WDOT(20) = (+ROP2(121) +ROP2(146) +ROP2(165) +ROP2(167)
            -ROP2(185) -ROP2(186) -ROP2(187) -ROP2(188)
            -ROP2(189) +ROP2(192) +ROP2(195) +ROP2(196)
            +ROP2(197) +ROP2(198) +ROP2(206))*rateconv *molwt[19];

    WDOT(22) = 0.0;
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RDWDOT3_THRD, RDWDOT3_BLK)
rdwdot3_kernel (const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt)
{

    real ROP114 =  ROP2(114) - ROP2(122);

    WDOT(13) = (+ROP2(31) +ROP2(37) +ROP2(41) +ROP2(53)
            +ROP2(55) +ROP2(63) +ROP2(70) -ROP2(71)
            -ROP2(72) -ROP2(73) -ROP2(74) -ROP2(75)
            -ROP2(76) -ROP2(77) +ROP2(79) +ROP2(83)
            -ROP2(90) +ROP2(96) +ROP2(99) +ROP2(100)
            +ROP2(101) +ROP2(140) +ROP2(154) +ROP2(160)
            +ROP2(172) +ROP2(177) +ROP2(188) +ROP2(201)
            +ROP2(204))*rateconv *molwt[12];
    WDOT(14) = (+ROP2(57) +ROP2(58) +ROP2(111) +ROP2(113)
            -ROP114 +ROP2(115) -ROP2(116) -ROP2(117)
            -ROP2(118) -ROP2(119) -ROP2(120) -ROP2(121)
            +ROP2(133) +ROP2(137) +ROP2(138) +ROP2(144))*rateconv *molwt[13];

    WDOT(15) = (+ROP2(91) +ROP2(92) +ROP2(95) +ROP2(105)
            +ROP2(132) +ROP2(142) +ROP2(143) -ROP2(155)
            -ROP2(156) -ROP2(157) -ROP2(158) -ROP2(159)
            -ROP2(160) -ROP2(161) -ROP2(162) -ROP2(163)
            -ROP2(164) -ROP2(165) -ROP2(166) -ROP2(167)
            -ROP2(168) -ROP2(169) +ROP2(171) +ROP2(174)
            +ROP2(176) +ROP2(191))*rateconv *molwt[14];
    WDOT(16) = (+ROP2(93) +ROP2(170) +ROP2(175) +ROP2(178)
            +ROP2(179) -ROP2(180) -ROP2(181) -ROP2(182)
            -ROP2(183) -ROP2(184))*rateconv *molwt[15];

    WDOT(17) = (+ROP2(39) -ROP2(95) -ROP2(108) -ROP2(109)
            -ROP2(110) -ROP2(111) -ROP2(112) -ROP2(113) -ROP2(113)
            +ROP2(116) +ROP2(127) +ROP2(129) +ROP2(131))*rateconv *molwt[16];
    WDOT(18) = (+ROP2(56) +ROP2(77) +ROP2(118) +ROP2(124)
            -ROP2(126) -ROP2(127) -ROP2(128) -ROP2(129)
            -ROP2(130) -ROP2(131) +ROP2(135) +ROP2(150)
            +ROP2(151) +ROP2(152) +ROP2(153) +ROP2(193))*rateconv *molwt[17];
    WDOT(19) = (+ROP2(89) +ROP2(148) +ROP2(163) +ROP2(173))*rateconv *molwt[18];
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RDWDOT6_THRD, RDWDOT6_BLK)
rdwdot6_kernel (const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt)
{

    WDOT(11) = (-ROP2(29) -ROP2(30) -ROP2(31) -ROP2(32)
            -ROP2(33) +ROP2(34) -ROP2(39) +ROP2(40)
            +ROP2(42) +ROP2(43) +ROP2(45) +ROP2(46)
            +ROP2(47) -ROP2(56) +ROP2(61) +ROP2(65)
            +ROP2(66) +ROP2(70) +ROP2(88) +ROP2(95)
            +ROP2(108) +ROP2(109) +ROP2(109) +ROP2(110) +ROP2(110)
            +ROP2(111) +ROP2(112) +ROP2(113) +ROP2(113)
            +ROP2(117) +ROP2(119) +ROP2(120) +ROP2(123)
            +ROP2(128) +ROP2(136) +ROP2(143) +ROP2(147)
            +ROP2(154) +ROP2(164) +ROP2(179) +ROP2(189))*rateconv *molwt[10];
    WDOT(12) = (+ROP2(29) +ROP2(30) +ROP2(32) +ROP2(33)
            -ROP2(40) +ROP2(44) +ROP2(52) -ROP2(70)
            +ROP2(125) +ROP2(130))*rateconv *molwt[11];
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RDWDOT7_THRD, RDWDOT7_BLK)
rdwdot7_kernel (const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt)
{

    WDOT(9) = (+ROP2(48) +ROP2(49) +ROP2(64) -ROP2(78)
            -ROP2(79) -ROP2(80) -ROP2(81) -ROP2(82)
            -ROP2(83) -ROP2(84) -ROP2(85) -ROP2(86)
            -ROP2(87) -ROP2(88) -ROP2(89) -ROP2(90)
            -ROP2(91) -ROP2(92) -ROP2(93) -ROP2(93)
            -ROP2(94) -ROP2(94) -ROP2(95) +ROP2(97)
            +ROP2(102) +ROP2(103) +ROP2(104) +ROP2(106) +ROP2(106)
            +ROP2(107) +ROP2(107) +ROP2(119) -ROP2(121)
            +ROP2(128) +ROP2(136) -ROP2(144) -ROP2(145)
            -ROP2(146) +ROP2(147) +ROP2(149) +ROP2(159)
            -ROP2(168) -ROP2(169) +ROP2(172) +ROP2(177)
            +ROP2(183) -ROP2(184) +ROP2(191) +ROP2(193)
            -ROP2(198) +ROP2(199) -ROP2(205) +ROP2(206))*rateconv *molwt[8];
    WDOT(10) = (+ROP2(78) +ROP2(84) +ROP2(86) +ROP2(88)
            +ROP2(90) -ROP2(102) -ROP2(103) -ROP2(104)
            -ROP2(105) -ROP2(106) -ROP2(107) +ROP2(144)
            +ROP2(166) +ROP2(168) +ROP2(184) +ROP2(186)
            +ROP2(198) +ROP2(205))*rateconv *molwt[9];

}

template <class real>
__global__ void
LAUNCH_BOUNDS (RDWDOT8_THRD, RDWDOT8_BLK)
rdwdot8_kernel (const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt)
{
    real ROP12 = ROP2(12) + ROP2(13) + ROP2(14)+ ROP2(15);
    real ROP22 = ROP2(22) + ROP2(23);
    real ROP27 = ROP2(27) + ROP2(28);

    WDOT(7) = (+ROP12 -ROP2(17) -ROP2(18) -ROP2(19)
            -ROP2(20) -ROP2(21) -ROP22 -ROP22
            +ROP2(24) +ROP2(26) +ROP27 -ROP2(33)
            +ROP2(47) -ROP2(55) +ROP2(75) -ROP2(76)
            -ROP2(84) -ROP2(85) +ROP2(86) +ROP2(101)
            +ROP2(138) -ROP2(141) +ROP2(142) +ROP2(153)
            +ROP2(162) -ROP2(163) +ROP2(174) -ROP2(175)
            -ROP2(176) -ROP2(177) +ROP2(178) -ROP2(187)
            -ROP2(188) -ROP2(197) +ROP2(203)
            -ROP2(204))*rateconv *molwt[6];
    WDOT(8) = (+ROP2(16) +ROP22 -ROP2(24) -ROP2(25)
            -ROP2(26) -ROP27 +ROP2(76) -ROP2(86)
            -ROP2(142) +ROP2(176) -ROP2(178)
            +ROP2(197))*rateconv *molwt[7];
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RDWDOT9_THRD, RDWDOT9_BLK)
rdwdot9_kernel (const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt)
{
    real ROP27 = ROP2(27) + ROP2(28);
    WDOT(5) = (+ROP2(1) +ROP2(2) -ROP2(3) -ROP2(4) -ROP2(4)
            -ROP2(9) +ROP2(10) -ROP2(16) -ROP2(16)
            +ROP2(19) +ROP2(19) +ROP2(20) -ROP2(21)
            +ROP2(25) +ROP2(26) -ROP27 -ROP2(30)
            +ROP2(33) -ROP2(35) +ROP2(43) -ROP2(45)
            +ROP2(51) -ROP2(53) -ROP2(54) +ROP2(55)
            -ROP2(63) +ROP2(65) +ROP2(73) -ROP2(74)
            -ROP2(80) -ROP2(81) +ROP2(83) +ROP2(85)
            +ROP2(97) +ROP2(99) -ROP2(100) +ROP2(103)
            -ROP2(104) +ROP2(110) -ROP2(118) -ROP2(119)
            -ROP2(124) +ROP2(129) -ROP2(131) -ROP2(137)
            +ROP2(141) +ROP2(151) -ROP2(152) +ROP2(154)
            +ROP2(158) -ROP2(161) +ROP2(163) +ROP2(177)
            +ROP2(181) -ROP2(182) +ROP2(188) +ROP2(195)
            -ROP2(196) -ROP2(202) +ROP2(204))*rateconv *molwt[4];
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RDWDOT10_THRD, RDWDOT10_BLK)
rdwdot10_kernel (const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt)
{

    real ROP12 = ROP2(12) + ROP2(13) + ROP2(14)+ ROP2(15);
    real ROP22 = ROP2(22) + ROP2(23);
    real ROP27 = ROP2(27) + ROP2(28);
    real ROP5 = ROP2(5) + ROP2(6) + ROP2(7) + ROP2(8);

    WDOT(1) = (-ROP2(2) -ROP2(3) +ROP5 +ROP2(18)
            +ROP2(24) -ROP2(31) -ROP2(36) +ROP2(42)
            -ROP2(49) +ROP2(58) +ROP2(60) +ROP2(61)
            -ROP2(64) +ROP2(72) +ROP2(96) +ROP2(102)
            +ROP2(127) +ROP2(133) +ROP2(134) +ROP2(150)
            +ROP2(155) +ROP2(157) +ROP2(171) +ROP2(180)
            +ROP2(192) +ROP2(200))*rateconv*molwt[0] ;

    WDOT(3) = (+ROP2(1) -ROP2(2) +ROP2(4) -ROP2(10)
            -ROP2(11) -ROP2(11) +ROP2(17) -ROP2(20)
            -ROP2(26) -ROP2(29) +ROP2(32) -ROP2(34)
            +ROP2(38) -ROP2(43) -ROP2(44) -ROP2(50)
            -ROP2(61) -ROP2(62) -ROP2(73) -ROP2(79)
            +ROP2(82) -ROP2(99) -ROP2(103) -ROP2(109)
            -ROP2(116) -ROP2(117) -ROP2(123) -ROP2(129)
            -ROP2(130) -ROP2(135) -ROP2(136) +ROP2(139)
            -ROP2(151) -ROP2(158) -ROP2(159) -ROP2(160)
            -ROP2(172) -ROP2(173) -ROP2(181) -ROP2(193)
            -ROP2(194) -ROP2(195) -ROP2(201))*rateconv *molwt[2];

    WDOT(4) = (-ROP2(1) +ROP2(11) -ROP12 +ROP2(18)
            +ROP2(20) +ROP2(21) +ROP22 -ROP2(32)
            -ROP2(38) -ROP2(47) -ROP2(51) -ROP2(52)
            -ROP2(65) -ROP2(66) -ROP2(75) -ROP2(82)
            -ROP2(83) +ROP2(84) -ROP2(101) -ROP2(110)
            -ROP2(125) -ROP2(138) -ROP2(139) -ROP2(140)
            -ROP2(153) -ROP2(154) -ROP2(162) -ROP2(174)
            +ROP2(175) +ROP2(187) -ROP2(203))*rateconv *molwt[3];
    WDOT(6) = (+ROP2(3) +ROP2(4) +ROP2(9) +ROP2(17)
            +ROP2(21) +ROP2(25) +ROP27 -ROP2(37)
            +ROP2(45) +ROP2(54) +ROP2(66) +ROP2(74)
            +ROP2(80) +ROP2(81) +ROP2(98) +ROP2(100)
            +ROP2(104) +ROP2(131) +ROP2(137) +ROP2(152)
            +ROP2(161) +ROP2(182) +ROP2(196) +ROP2(202))*rateconv *molwt[5];
}



#endif //RDWDOT_H
