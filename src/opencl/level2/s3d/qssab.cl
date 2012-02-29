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
qssab_kernel(__global real* RF, __global real* RB, __global real* A)
{

    real DEN;

    A(8,0) = A(8,0) + A(8,10)*A(10,0);
    DEN = 1 -A(8,10)*A(10,8);
    A(8,0) = DIV (A(8,0), DEN);
    A(8,4) = DIV (A(8,4), DEN);
    A(8,3) = DIV (A(8,3), DEN);
    A(3,0) = A(3,0) + A(3,5)*A(5,0);
    DEN = 1 -A(3,5)*A(5,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(3,1) = DIV (A(3,1), DEN);
    A(3,8) = DIV (A(3,8), DEN);
    A(3,6) = DIV (A(3,6), DEN);
    A(4,0) = A(4,0) + A(4,9)*A(9,0);
    A(4,7) = A(4,7) + A(4,9)*A(9,7);
    DEN = 1 -A(4,9)*A(9,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(4,2) = DIV (A(4,2), DEN);
    A(4,1) = DIV (A(4,1), DEN);
    A(4,8) = DIV (A(4,8), DEN);
    A(7,0) = A(7,0) + A(7,9)*A(9,0);
    A(7,4) = A(7,4) + A(7,9)*A(9,4);
    DEN = 1 -A(7,9)*A(9,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,2) = DIV (A(7,2), DEN);
    A(7,1) = DIV (A(7,1), DEN);
    A(7,6) = DIV (A(7,6), DEN);
    A(3,0) = A(3,0) + A(3,6)*A(6,0);
    A(3,7) = A(3,6)*A(6,7);
    A(3,2) = A(3,2) + A(3,6)*A(6,2);
    DEN = 1 -A(3,6)*A(6,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(3,1) = DIV (A(3,1), DEN);
    A(3,8) = DIV (A(3,8), DEN);
    A(7,0) = A(7,0) + A(7,6)*A(6,0);
    A(7,3) = A(7,6)*A(6,3);
    A(7,2) = A(7,2) + A(7,6)*A(6,2);
    DEN = 1 -A(7,6)*A(6,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,3) = DIV (A(7,3), DEN);
    A(7,2) = DIV (A(7,2), DEN);
    A(7,1) = DIV (A(7,1), DEN);
    A(2,0) = A(2,0) + A(2,6)*A(6,0);
    A(2,3) = A(2,3) + A(2,6)*A(6,3);
    A(2,7) = A(2,7) + A(2,6)*A(6,7);
    DEN = 1 -A(2,6)*A(6,2);
    A(2,0) = DIV (A(2,0), DEN);
    A(2,4) = DIV (A(2,4), DEN);
    A(2,3) = DIV (A(2,3), DEN);
    A(2,7) = DIV (A(2,7), DEN);
    A(2,1) = DIV (A(2,1), DEN);
    A(4,0) = A(4,0) + A(4,8)*A(8,0);
    A(4,3) = A(4,3) + A(4,8)*A(8,3);
    DEN = 1 -A(4,8)*A(8,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(4,2) = DIV (A(4,2), DEN);
    A(4,1) = DIV (A(4,1), DEN);
    A(3,0) = A(3,0) + A(3,8)*A(8,0);
    A(3,4) = A(3,4) + A(3,8)*A(8,4);
    DEN = 1 -A(3,8)*A(8,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(3,1) = DIV (A(3,1), DEN);
    A(4,0) = A(4,0) + A(4,1)*A(1,0);
    A(4,3) = A(4,3) + A(4,1)*A(1,3);
    A(4,7) = A(4,7) + A(4,1)*A(1,7);
    A(4,2) = A(4,2) + A(4,1)*A(1,2);
    DEN = 1 -A(4,1)*A(1,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(4,2) = DIV (A(4,2), DEN);
    A(3,0) = A(3,0) + A(3,1)*A(1,0);
    A(3,4) = A(3,4) + A(3,1)*A(1,4);
    A(3,7) = A(3,7) + A(3,1)*A(1,7);
    A(3,2) = A(3,2) + A(3,1)*A(1,2);
    DEN = 1 -A(3,1)*A(1,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(7,0) = A(7,0) + A(7,1)*A(1,0);
    A(7,4) = A(7,4) + A(7,1)*A(1,4);
    A(7,3) = A(7,3) + A(7,1)*A(1,3);
    A(7,2) = A(7,2) + A(7,1)*A(1,2);
    DEN = 1 -A(7,1)*A(1,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,3) = DIV (A(7,3), DEN);
    A(7,2) = DIV (A(7,2), DEN);
    A(2,0) = A(2,0) + A(2,1)*A(1,0);
    A(2,4) = A(2,4) + A(2,1)*A(1,4);
    A(2,3) = A(2,3) + A(2,1)*A(1,3);
    A(2,7) = A(2,7) + A(2,1)*A(1,7);
    DEN = 1 -A(2,1)*A(1,2);
    A(2,0) = DIV (A(2,0), DEN);
    A(2,4) = DIV (A(2,4), DEN);
    A(2,3) = DIV (A(2,3), DEN);
    A(2,7) = DIV (A(2,7), DEN);
    A(4,0) = A(4,0) + A(4,2)*A(2,0);
    A(4,3) = A(4,3) + A(4,2)*A(2,3);
    A(4,7) = A(4,7) + A(4,2)*A(2,7);
    DEN = 1 -A(4,2)*A(2,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(3,0) = A(3,0) + A(3,2)*A(2,0);
    A(3,4) = A(3,4) + A(3,2)*A(2,4);
    A(3,7) = A(3,7) + A(3,2)*A(2,7);
    DEN = 1 -A(3,2)*A(2,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(7,0) = A(7,0) + A(7,2)*A(2,0);
    A(7,4) = A(7,4) + A(7,2)*A(2,4);
    A(7,3) = A(7,3) + A(7,2)*A(2,3);
    DEN = 1 -A(7,2)*A(2,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,3) = DIV (A(7,3), DEN);
    A(4,0) = A(4,0) + A(4,7)*A(7,0);
    A(4,3) = A(4,3) + A(4,7)*A(7,3);
    DEN = 1 -A(4,7)*A(7,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(3,0) = A(3,0) + A(3,7)*A(7,0);
    A(3,4) = A(3,4) + A(3,7)*A(7,4);
    DEN = 1 -A(3,7)*A(7,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(4,0) = A(4,0) + A(4,3)*A(3,0);
    DEN = 1 -A(4,3)*A(3,4);
    A(4,0) = DIV (A(4,0), DEN);
}
