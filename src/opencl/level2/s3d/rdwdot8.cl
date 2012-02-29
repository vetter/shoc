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
rdwdot8_kernel (__global const real* RKF, __global const real* RKR,
		__global real* WDOT, const real rateconv, __global const real* molwt)
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
