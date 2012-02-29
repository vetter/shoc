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
rdwdot_kernel (__global const real* RKF, __global const real* RKR,
		__global real* WDOT, const real rateconv, __global const real* molwt)
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
