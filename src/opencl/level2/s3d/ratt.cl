#ifdef K_DOUBLE_PRECISION
#define DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif AMD_DOUBLE_PRECISION
#define DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif

// Macros to explicitly control precision of the constants, otherwise
// known to cause problems for some Compilers
#ifdef DOUBLE_PRECISION
#define CPREC(a) a
#else
#define CPREC(a) a##f
#endif

//replace divisions by multiplication with the reciprocal
#define REPLACE_DIV_WITH_RCP 1

//Call the appropriate math function based on precision
#ifdef DOUBLE_PRECISION
#define real double
#if REPLACE_DIV_WITH_RCP
#define DIV(x,y) ((x)*(1.0/(y)))
#else
#define DIV(x,y) ((x)/(y))
#endif
#define POW pow
#define EXP exp
#define EXP10 exp10
#define EXP2 exp2
#define MAX fmax
#define MIN fmin
#define LOG log
#define LOG10 log10
#else
#define real float
#if REPLACE_DIV_WITH_RCP
#define DIV(x,y) ((x)*(1.0f/(y)))
#else
#define DIV(x,y) ((x)/(y))
#endif
#define POW pow
#define EXP exp
#define EXP10 exp10
#define EXP2 exp2
#define MAX fmax
#define MIN fmin
#define LOG log
#define LOG10 log10
#endif

//Kernel indexing macros
#define thread_num (get_global_id(0))
#define idx2(p,z) (p[(((z)-1)*(N_GP)) + thread_num])
#define idx(x, y) ((x)[(y)-1])
#define C(q)     idx2(C, q)
#define Y(q)     idx2(Y, q)
#define RF(q)    idx2(RF, q)
#define EG(q)    idx2(EG, q)
#define RB(q)    idx2(RB, q)
#define RKLOW(q) idx2(RKLOW, q)
#define ROP(q)   idx(ROP, q)
#define WDOT(q)  idx2(WDOT, q)
#define RKF(q)   idx2(RKF, q)
#define RKR(q)   idx2(RKR, q)
#define A_DIM    (11)
#define A(b, c)  idx2(A, (((b)*A_DIM)+c) )


__kernel void
ratt_kernel(__global const real* T, __global real* RF, real TCONV)
{
       
    const real TEMP = T[get_global_id(0)]*TCONV;

    const real ALOGT = LOG(TEMP);
    const real TI = 1.0e0/(TEMP);
    const real TI2 = TI*TI;
    real TMP;

    RF(1) = EXP(3.20498617e1 -7.25286183e3*TI);
    RF(2) = EXP(1.08197783e1 +2.67e0*ALOGT -3.16523284e3*TI);
    RF(3) = EXP(1.9190789e1 +1.51e0*ALOGT -1.72603317e3*TI);
    RF(4) = EXP(1.0482906e1 +2.4e0*ALOGT +1.06178717e3*TI);
    RF(5) = 1.e18*TI;
    RF(6) = EXP(3.90385861e1 -6.e-1*ALOGT);
    RF(7) = EXP(4.55408762e1 -1.25e0*ALOGT);
    RF(8) = 5.5e20*TI2;
    RF(9) = 2.2e22*TI2;
    RF(10) = 5.e17*TI;
    RF(11) = 1.2e17*TI;
    RF(12) = EXP(4.24761511e1 -8.6e-1*ALOGT);
    RF(13) = EXP(4.71503141e1 -1.72e0*ALOGT);
    RF(14) = EXP(4.42511034e1 -7.6e-1*ALOGT);
    RF(15) = EXP(4.47046282e1 -1.24e0*ALOGT);
    RF(16) = EXP(3.19350862e1 -3.7e-1*ALOGT);
    RF(17) = EXP(2.90097872e1 -3.37658384e2*TI);
    RF(18) = EXP(3.04404238e1 -4.12637667e2*TI);
    RF(19) = EXP(3.18908801e1 -1.50965e2*TI);
    RF(20) = 2.e13;
    RF(21) = EXP(3.14683206e1 +2.51608334e2*TI);
    RF(22) = EXP(2.55908003e1 +8.20243168e2*TI);
    RF(23) = EXP(3.36712758e1 -6.03860001e3*TI);
    RF(24) = EXP(1.6308716e1 +2.e0*ALOGT -2.61672667e3*TI);
    RF(25) = EXP(2.99336062e1 -1.81158e3*TI);
    RF(26) = EXP(1.60803938e1 +2.e0*ALOGT -2.01286667e3*TI);
    RF(27) = EXP(2.81906369e1 -1.61029334e2*TI);
    RF(28) = EXP(3.39940492e1 -4.81075134e3*TI);
    RF(29) = EXP(3.40312786e1 -1.50965e3*TI);
    RF(30) = EXP(1.76783433e1 +1.228e0*ALOGT -3.52251667e1*TI);
    RF(31) = EXP(1.75767107e1 +1.5e0*ALOGT -4.00560467e4*TI);
    RF(32) = EXP(2.85473118e1 -2.40537567e4*TI);
    RF(33) = EXP(3.26416564e1 -1.18759134e4*TI);
    RF(34) = 5.7e13;
    RF(35) = 3.e13;
    RF(36) = EXP(1.85223344e1 +1.79e0*ALOGT -8.40371835e2*TI);
    RF(37) = EXP(2.93732401e1 +3.79928584e2*TI);
    RF(38) = 3.3e13;
    RF(39) = 5.e13;
    RF(40) = EXP(2.88547965e1 -3.47219501e2*TI);
    RF(41) = EXP(2.77171988e1 +4.8e-1*ALOGT +1.30836334e2*TI);
    RF(42) = 7.34e13;
    RF(43) = 3.e13;
    RF(44) = 3.e13;
    RF(45) = 5.e13;
    RF(46) = EXP(3.9769885e1 -1.e0*ALOGT -8.55468335e3*TI);
    RF(47) = EXP(2.96591694e1 -2.01286667e2*TI);
    RF(48) = EXP(3.77576522e1 -8.e-1*ALOGT);
    RF(49) = EXP(1.31223634e1 +2.e0*ALOGT -3.63825651e3*TI);
    RF(50) = 8.e13;
    TMP = EXP(-7.54825001e2*TI);
    RF(51) = 1.056e13 * TMP;
    RF(52) = 2.64e12 * TMP;
    RF(53) = 2.e13;
    RF(54) = EXP(1.62403133e1 +2.e0*ALOGT -1.50965e3*TI);
    RF(55) = 2.e13;
    RF(56) = EXP(2.74203001e1 +5.e-1*ALOGT -2.26950717e3*TI);
    RF(57) = 4.e13;
    RF(58) = 3.2e13;
    RF(59) = EXP(3.03390713e1 -3.01930001e2*TI);
    RF(60) = 3.e13;
    RF(61) = 1.5e13;
    RF(62) = 1.5e13;
    RF(63) = 3.e13;
    RF(64) = 7.e13;
    RF(65) = 2.8e13;
    RF(66) = 1.2e13;
    RF(67) = 3.e13;
    RF(68) = 9.e12;
    RF(69) = 7.e12;
    RF(70) = 1.4e13;
    RF(71) = EXP(2.7014835e1 +4.54e-1*ALOGT -1.30836334e3*TI);
    RF(72) = EXP(2.38587601e1 +1.05e0*ALOGT -1.64803459e3*TI);
    RF(73) = EXP(3.12945828e1 -1.781387e3*TI);
    RF(74) = EXP(2.19558261e1 +1.18e0*ALOGT +2.2493785e2*TI);
    RF(75) = EXP(3.22361913e1 -2.01286667e4*TI);
    TMP = EXP(-4.02573334e3*TI);
    RF(76) = 1.e12 * TMP;
    RF(127) = 5.e13 * TMP;
    RF(129) = 1.e13 * TMP;
    RF(77) = EXP(3.21806786e1 +2.59156584e2*TI);
    RF(78) = EXP(3.70803784e1 -6.3e-1*ALOGT -1.92731984e2*TI);
    RF(79) = 8.43e13;
    RF(80) = EXP(1.78408622e1 +1.6e0*ALOGT -2.72743434e3*TI);
    RF(81) = 2.501e13;
    RF(82) = EXP(3.10595094e1 -1.449264e4*TI);
    RF(83) = EXP(2.43067848e1 -4.49875701e3*TI);
    RF(84) = 1.e12;
    RF(85) = 1.34e13;
    RF(86) = EXP(1.01064284e1 +2.47e0*ALOGT -2.60666234e3*TI);
    RF(87) = 3.e13;
    RF(88) = 8.48e12;
    RF(89) = 1.8e13;
    RF(90) = EXP(8.10772006e0 +2.81e0*ALOGT -2.94884967e3*TI);
    RF(91) = 4.e13;
    TMP = EXP(2.86833501e2*TI);
    RF(92) = 1.2e13 * TMP;
    RF(107) = 1.6e13 * TMP;
    RF(93) = EXP(3.75927776e1 -9.7e-1*ALOGT -3.11994334e2*TI);
    RF(94) = EXP(2.9238457e1 +1.e-1*ALOGT -5.33409668e3*TI);
    RF(95) = 5.e13;
    RF(96) = 2.e13;
    RF(97) = 3.2e13;
    RF(98) = 1.6e13;
    RF(99) = 1.e13;
    RF(100) = 5.e12;
    RF(101) = EXP(-2.84796532e1 +7.6e0*ALOGT +1.77635484e3*TI);
    RF(102) = EXP(2.03077504e1 +1.62e0*ALOGT -5.45486868e3*TI);
    RF(103) = EXP(2.07430685e1 +1.5e0*ALOGT -4.32766334e3*TI);
    RF(104) = EXP(1.84206807e1 +1.6e0*ALOGT -1.570036e3*TI);
    RF(105) = 6.e13;
    RF(106) = EXP(1.47156719e1 +2.e0*ALOGT -4.16160184e3*TI);
    RF(108) = 1.e14;
    RF(109) = 1.e14;
    RF(110) = EXP(2.81010247e1 -4.29747034e2*TI);
    RF(111) = 5.e13;
    RF(112) = 3.e13;
    RF(113) = 1.e13;
    RF(114) = EXP(3.43156328e1 -5.2e-1*ALOGT -2.55382459e4*TI);
    RF(115) = EXP(1.97713479e1 +1.62e0*ALOGT -1.86432818e4*TI);
    TMP = EXP(2.e0*ALOGT -9.56111669e2*TI );
    RF(116) = 1.632e7 * TMP;
    RF(117) = 4.08e6 * TMP;
    RF(118) = EXP(-8.4310155e0 +4.5e0*ALOGT +5.03216668e2*TI);
    RF(119) = EXP(-7.6354939e0 +4.e0*ALOGT +1.00643334e3*TI);
    RF(120) = EXP(1.61180957e1 +2.e0*ALOGT -3.01930001e3*TI);
    RF(121) = EXP(1.27430637e2 -1.182e1*ALOGT -1.79799315e4*TI);
    RF(122) = 1.e14;
    RF(123) = 1.e14;
    RF(124) = 2.e13;
    RF(125) = 1.e13;
    RF(126) = EXP(3.34301138e1 -6.e-2*ALOGT -4.27734167e3*TI);
    RF(128) = EXP(2.11287309e1 +1.43e0*ALOGT -1.35365284e3*TI);
    RF(130) = EXP(2.81906369e1 -6.79342501e2*TI);
    TMP = EXP(-1.00643334e3*TI);
    RF(131) = 7.5e12 * TMP;
    RF(152) = 1.e13 * TMP;
    RF(186) = 2.e13 * TMP;
    RF(132) = EXP(2.94360258e1 +2.7e-1*ALOGT -1.40900667e2*TI);
    RF(133) = 3.e13;
    RF(134) = 6.e13;
    RF(135) = 4.8e13;
    RF(136) = 4.8e13;
    RF(137) = 3.011e13;
    RF(138) = EXP(1.41081802e1 +1.61e0*ALOGT +1.9293327e2*TI);
    RF(139) = EXP(2.64270483e1 +2.9e-1*ALOGT -5.53538334e0*TI);
    RF(140) = EXP(3.83674178e1 -1.39e0*ALOGT -5.08248834e2*TI);
    RF(141) = 1.e13;
    RF(142) = EXP(2.32164713e1 +2.99917134e2*TI);
    RF(143) = 9.033e13;
    RF(144) = 3.92e11;
    RF(145) = 2.5e13;
    RF(146) = EXP(5.56675073e1 -2.83e0*ALOGT -9.36888792e3*TI);
    RF(147) = EXP(9.64601125e1 -9.147e0*ALOGT -2.36008617e4*TI);
    RF(148) = 1.e14;
    RF(149) = 9.e13;
    TMP = EXP(-2.01286667e3*TI);
    RF(150) = 2.e13 * TMP;
    RF(151) = 2.e13 * TMP;
    RF(153) = 1.4e11;
    RF(154) = 1.8e10;
    RF(155) = EXP(2.97104627e1 +4.4e-1*ALOGT -4.46705436e4*TI);
    RF(156) = EXP(2.77079822e1 +4.54e-1*ALOGT -9.15854335e2*TI);
    RF(157) = EXP(1.77414365e1 +1.93e0*ALOGT -6.51665585e3*TI);
    RF(158) = EXP(1.65302053e1 +1.91e0*ALOGT -1.88203034e3*TI);
    TMP = EXP(1.83e0*ALOGT -1.10707667e2*TI );
    RF(159) = 1.92e7 * TMP;
    RF(160) = 3.84e5 * TMP;
    RF(161) = EXP(1.50964444e1 +2.e0*ALOGT -1.25804167e3*TI);
    RF(162) = EXP(3.13734413e1 -3.05955734e4*TI);
    RF(163) = EXP(2.83241683e1 -7.04503335e3*TI);
    RF(164) = EXP(1.61180957e1 +2.e0*ALOGT -4.02573334e3*TI);
    RF(165) = EXP(3.06267534e1 -3.01930001e3*TI);
    RF(166) = 5.e13;
    RF(167) = 5.e13;
    RF(168) = EXP(1.23327053e1 +2.e0*ALOGT -4.62959334e3*TI);
    RF(169) = EXP(2.65223585e1 -3.87476834e3*TI);
    RF(170) = EXP(4.07945264e1 -9.9e-1*ALOGT -7.95082335e2*TI);
    RF(171) = 2.e12;
    RF(172) = 1.604e13;
    RF(173) = 8.02e13;
    RF(174) = 2.e10;
    RF(175) = 3.e11;
    RF(176) = 3.e11;
    RF(177) = 2.4e13;
    RF(178) = EXP(2.28865889e1 -4.90133034e2*TI);
    RF(179) = 1.2e14;
    RF(180) = EXP(1.85604427e1 +1.9e0*ALOGT -3.78922151e3*TI);
    RF(181) = EXP(1.83130955e1 +1.92e0*ALOGT -2.86330284e3*TI);
    RF(182) = EXP(1.50796373e1 +2.12e0*ALOGT -4.37798501e2*TI);
    RF(183) = EXP(3.13199006e1 +2.76769167e2*TI);
    RF(184) = EXP(1.56303353e1 +1.74e0*ALOGT -5.25861418e3*TI);
    RF(185) = 2.e14;
    RF(187) = 2.66e12;
    RF(188) = 6.6e12;
    RF(189) = 6.e13;
    RF(190) = EXP(3.02187852e1 -1.64083859e3*TI);
    RF(191) = EXP(5.11268757e1 -2.39e0*ALOGT -5.62596234e3*TI);
    RF(192) = EXP(1.20435537e1 +2.5e0*ALOGT -1.2530095e3*TI);
    RF(193) = EXP(1.86030023e1 +1.65e0*ALOGT -1.6455185e2*TI);
    RF(194) = EXP(1.73708586e1 +1.65e0*ALOGT +4.89126601e2*TI);
    RF(195) = EXP(2.59162227e1 +7.e-1*ALOGT -2.95891401e3*TI);
    RF(196) = EXP(1.49469127e1 +2.e0*ALOGT +1.49958567e2*TI);
    RF(197) = EXP(9.16951838e0 +2.6e0*ALOGT -6.99974385e3*TI);
    RF(198) = EXP(7.8845736e-1 +3.5e0*ALOGT -2.85575459e3*TI);
    RF(199) = EXP(5.65703751e1 -2.92e0*ALOGT -6.29272443e3*TI);
    RF(200) = 1.8e12;
    RF(201) = 9.6e13;
    RF(202) = 2.4e13;
    RF(203) = 9.e10;
    RF(204) = 2.4e13;
    RF(205) = 1.1e13;
    RF(206) = EXP(7.50436995e1 -5.22e0*ALOGT -9.93701954e3*TI);
}
