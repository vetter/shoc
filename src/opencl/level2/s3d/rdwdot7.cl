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
rdwdot7_kernel (__global const real* RKF, __global const real* RKR,
		__global real* WDOT, const real rateconv, __global const real* molwt)
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
