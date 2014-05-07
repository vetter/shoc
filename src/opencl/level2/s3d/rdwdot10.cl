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
rdwdot10_kernel (__global const real* RKF, __global const real* RKR,
		__global real* WDOT, const real rateconv, __global const real* molwt)
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
