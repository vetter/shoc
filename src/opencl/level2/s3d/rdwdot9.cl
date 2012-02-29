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
rdwdot9_kernel (__global const real* RKF, __global const real* RKR,
		__global real* WDOT, const real rateconv, __global const real* molwt)
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
