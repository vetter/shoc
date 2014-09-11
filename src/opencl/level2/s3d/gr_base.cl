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
gr_base(__global const real* P, __global const real* T, __global const real* Y,
		__global real* C, const real TCONV, const real PCONV)
{

    const real TEMP = T[get_global_id(0)]*TCONV;
    const real PRES = P[get_global_id(0)]*PCONV;

#ifdef DOUBLE_PRECISION
    const real SMALL = CPREC(1.0e-50);
#else
    const real SMALL = FLT_MIN;
#endif

    real SUM, ctmp;

    SUM = 0.0;

    C(1)  = ctmp = Y(1) *CPREC(4.96046521e-1);
    SUM  += ctmp;
    C(2)  = ctmp = Y(2) *CPREC(9.92093043e-1);
    SUM  += ctmp;
    C(3)  = ctmp = Y(3) *CPREC(6.25023433e-2);
    SUM  += ctmp;
    C(4)  = ctmp = Y(4) *CPREC(3.12511716e-2);
    SUM  += ctmp;
    C(5)  = ctmp = Y(5) *CPREC(5.87980383e-2);
    SUM  += ctmp;
    C(6)  = ctmp = Y(6) *CPREC(5.55082499e-2);
    SUM  += ctmp;
    C(7)  = ctmp = Y(7) *CPREC(3.02968146e-2);
    SUM  += ctmp;
    C(8)  = ctmp = Y(8) *CPREC(2.93990192e-2);
    SUM  += ctmp;
    C(9)  = ctmp = Y(9) *CPREC(6.65112065e-2);
    SUM  += ctmp;
    C(10) = ctmp = Y(10)*CPREC(6.23323639e-2);
    SUM  += ctmp;
    C(11) = ctmp = Y(11)*CPREC(3.57008335e-2);
    SUM  += ctmp;
    C(12) = ctmp = Y(12)*CPREC(2.27221341e-2);
    SUM  += ctmp;
    C(13) = ctmp = Y(13)*CPREC(3.33039255e-2);
    SUM  += ctmp;
    C(14) = ctmp = Y(14)*CPREC(3.84050525e-2);
    SUM  += ctmp;
    C(15) = ctmp = Y(15)*CPREC(3.56453112e-2);
    SUM  += ctmp;
    C(16) = ctmp = Y(16)*CPREC(3.32556033e-2);
    SUM  += ctmp;
    C(17) = ctmp = Y(17)*CPREC(2.4372606e-2);
    SUM  += ctmp;
    C(18) = ctmp = Y(18)*CPREC(2.37882046e-2);
    SUM  += ctmp;
    C(19) = ctmp = Y(19)*CPREC(2.26996304e-2);
    SUM  += ctmp;
    C(20) = ctmp = Y(20)*CPREC(2.43467162e-2);
    SUM  += ctmp;
    C(21) = ctmp = Y(21)*CPREC(2.37635408e-2);
    SUM  += ctmp;
    C(22) = ctmp = Y(22)*CPREC(3.56972032e-2);
    SUM  += ctmp;

    SUM = DIV (PRES, (SUM * (TEMP) * CPREC(8.314510e7)));

// #pragma unroll 22
    for (unsigned k=1; k<=22; k++)
    {
        C(k) = MAX(C(k), SMALL) * SUM;
    }
}
