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
ratt4_kernel(__global const real* T, __global const real* RF, __global real* RB,
		__global const real* EG, const real TCONV)
{

    real TEMP = T[get_global_id(0)]*TCONV;
    real ALOGT = LOG(TEMP);
#ifdef DOUBLE_PRECISION
    const real SMALL_INV = 1e+300;
#else
    const real SMALL_INV = 1e+20f;
#endif

    const real RU=CPREC(8.31451e7);
    const real PATM = CPREC(1.01325e6);
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    real rtemp_inv;

    rtemp_inv = DIV ((EG(4)*EG(10)), (EG(5)*EG(16)));
    RB(51) = RF(51) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(10)), (EG(2)*EG(2)*EG(15)*PFAC));
    RB(52) = RF(52) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(10)), (EG(2)*EG(17)));
    RB(53) = RF(53) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(10)), (EG(6)*EG(9)));
    RB(54) = RF(54) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(10)), (EG(5)*EG(17)));
    RB(55) = RF(55) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(14)*PFAC), EG(26));
    RB(56) = RF(56) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(10)), (EG(2)*EG(19)));
    RB(57) = RF(57) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(10)), (EG(1)*EG(19)));
    RB(58) = RF(58) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(59) = RF(59) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(11)), (EG(1)*EG(9)));
    RB(60) = RF(60) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(11)), (EG(1)*EG(14)));
    RB(61) = RF(61) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(11)), (EG(2)*EG(16)));
    RB(62) = RF(62) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(11)), (EG(2)*EG(17)));
    RB(63) = RF(63) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(11)), (EG(2)*EG(12)));
    RB(64) = RF(64) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(11)), (EG(2)*EG(5)*EG(14)*PFAC));
    RB(65) = RF(65) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(11)), (EG(6)*EG(14)));
    RB(66) = RF(66) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(67) = RF(67) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(68) = RF(68) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(69) = RF(69) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(15)), (EG(14)*EG(17)));
    RB(70) = RF(70) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(17)*PFAC), EG(18));
    RB(71) = RF(71) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(17)), (EG(1)*EG(16)));
    RB(72) = RF(72) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(17)), (EG(5)*EG(16)));
    RB(73) = RF(73) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(17)), (EG(6)*EG(16)));
    RB(74) = RF(74) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(17)), (EG(7)*EG(16)));
    RB(75) = RF(75) * MIN(rtemp_inv, SMALL_INV);
}
