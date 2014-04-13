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
rdwdot6_kernel (__global const real* RKF, __global const real* RKR,
		__global real* WDOT, const real rateconv, __global const real* molwt)
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
