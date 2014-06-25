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

#define ROP2(a)  (RKF(a) - RKR (a))


__kernel void
rdwdot3_kernel (__global const real* RKF, __global const real* RKR,
		__global real* WDOT, const real rateconv, __global const real* molwt)
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
