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
ratt6_kernel(__global const real* T, __global const real* RF,
		__global real* RB, __global const real* EG, const real TCONV)
{

    const real TEMP = T[get_global_id(0)]*TCONV;
    const real ALOGT = LOG(TEMP);
#ifdef DOUBLE_PRECISION
    const real SMALL_INV = 1e+300;
#else
    const real SMALL_INV = 1e+20f;
#endif

    const real RU=CPREC(8.31451e7);
    const real PATM = CPREC(1.01325e6);
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    real rtemp_inv;

    rtemp_inv = DIV ((EG(4)*EG(18)), (EG(7)*EG(17)));
    RB(101) = RF(101) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(13)), (EG(1)*EG(12)));
    RB(102) = RF(102) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(13)), (EG(5)*EG(12)));
    RB(103) = RF(103) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(13)), (EG(6)*EG(12)));
    RB(104) = RF(104) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(13)), (EG(2)*EG(22)));
    RB(105) = RF(105) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(13)), (EG(12)*EG(12)));
    RB(106) = RF(106) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(13)), (EG(12)*EG(12)));
    RB(107) = RF(107) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(25)), (EG(11)*EG(14)));
    RB(108) = RF(108) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(25)), (EG(2)*EG(14)*EG(14)*PFAC));
    RB(109) = RF(109) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(25)), (EG(5)*EG(14)*EG(14)*PFAC));
    RB(110) = RF(110) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(25)), (EG(14)*EG(19)));
    RB(111) = RF(111) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(25)), (EG(14)*EG(21)));
    RB(112) = RF(112) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(25)*EG(25)), (EG(14)*EG(14)*EG(19)*PFAC));
    RB(113) = RF(113) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(19), EG(20));
    RB(114) = RF(114) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(21), (EG(2)*EG(19)*PFAC));
    RB(115) = RF(115) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(19)), (EG(2)*EG(25)));
    RB(116) = RF(116) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(19)), (EG(10)*EG(14)));
    RB(117) = RF(117) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(19)), (EG(2)*EG(26)));
    RB(118) = RF(118) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(19)), (EG(12)*EG(14)));
    RB(119) = RF(119) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(19)), (EG(14)*EG(21)));
    RB(120) = RF(120) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(19)*PFAC), EG(29));
    RB(121) = RF(121) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(19), EG(20));
    RB(122) = RF(122) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(20)), (EG(10)*EG(14)));
    RB(123) = RF(123) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(20)), (EG(2)*EG(26)));
    RB(124) = RF(124) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(20)), (EG(10)*EG(15)));
    RB(125) = RF(125) * MIN(rtemp_inv, SMALL_INV);
}
