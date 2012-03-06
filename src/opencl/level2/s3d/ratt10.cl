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
ratt10_kernel(__global const real* T, __global real* RKLOW, real TCONV)
{

    const real TEMP = T[get_global_id(0)]*TCONV;
    const real ALOGT = LOG(TEMP);

    RKLOW(1) = EXP(CPREC(4.22794408e1) -CPREC(9.e-1)*ALOGT + DIV(CPREC(8.55468335e2),TEMP));
    RKLOW(2) = EXP(CPREC(6.37931383e1) -CPREC(3.42e0)*ALOGT - DIV(CPREC(4.24463259e4),TEMP));
    RKLOW(3) = EXP(CPREC(6.54619238e1) -CPREC(3.74e0)*ALOGT - DIV(CPREC(9.74227469e2),TEMP));
    RKLOW(4) = EXP(CPREC(5.55621468e1) -CPREC(2.57e0)*ALOGT - DIV(CPREC(7.17083751e2),TEMP));
    RKLOW(5) = EXP(CPREC(6.33329483e1) -CPREC(3.14e0)*ALOGT - DIV(CPREC(6.18956501e2),TEMP));
    RKLOW(6) = EXP(CPREC(7.69748493e1) -CPREC(5.11e0)*ALOGT - DIV(CPREC(3.57032226e3),TEMP));
    RKLOW(7) = EXP(CPREC(6.98660102e1) -CPREC(4.8e0)*ALOGT - DIV(CPREC(2.79788467e3),TEMP));
    RKLOW(8) = EXP(CPREC(7.68923562e1) -CPREC(4.76e0)*ALOGT - DIV(CPREC(1.22784867e3),TEMP));
    RKLOW(9) = EXP(CPREC(1.11312542e2) -CPREC(9.588e0)*ALOGT - DIV(CPREC(2.566405e3),TEMP));
    RKLOW(10) = EXP(CPREC(1.15700234e2) -CPREC(9.67e0)*ALOGT - DIV(CPREC(3.13000767e3),TEMP));
    RKLOW(11) = EXP(CPREC(3.54348644e1) -CPREC(6.4e-1)*ALOGT - DIV(CPREC(2.50098684e4),TEMP));
    RKLOW(12) = EXP(CPREC(6.3111756e1) -CPREC(3.4e0)*ALOGT - DIV(CPREC(1.80145126e4),TEMP));
    RKLOW(13) = EXP(CPREC(9.57409899e1) -CPREC(7.64e0)*ALOGT - DIV(CPREC(5.98827834e3),TEMP));
    RKLOW(14) = EXP(CPREC(6.9414025e1) -CPREC(3.86e0)*ALOGT - DIV(CPREC(1.67067934e3),TEMP));
    RKLOW(15) = EXP(CPREC(1.35001549e2) -CPREC(1.194e1)*ALOGT - DIV(CPREC(4.9163262e3),TEMP));
    RKLOW(16) = EXP(CPREC(9.14494773e1) -CPREC(7.297e0)*ALOGT - DIV(CPREC(2.36511834e3),TEMP));
    RKLOW(17) = EXP(CPREC(1.17075165e2) -CPREC(9.31e0)*ALOGT - DIV(CPREC(5.02512164e4),TEMP));
    RKLOW(18) = EXP(CPREC(9.68908955e1) -CPREC(7.62e0)*ALOGT - DIV(CPREC(3.50742017e3),TEMP));
    RKLOW(19) = EXP(CPREC(9.50941235e1) -CPREC(7.08e0)*ALOGT - DIV(CPREC(3.36400342e3),TEMP));
    RKLOW(20) = EXP(CPREC(1.38440285e2) -CPREC(1.2e1)*ALOGT - DIV(CPREC(3.00309643e3),TEMP));
    RKLOW(21) = EXP(CPREC(8.93324137e1) -CPREC(6.66e0)*ALOGT - DIV(CPREC(3.52251667e3),TEMP));
}
