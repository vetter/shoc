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
ratt8_kernel(__global const real* T, __global const real* RF,
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

    rtemp_inv = DIV ((EG(3)*EG(27)), (EG(5)*EG(26)));
    RB(151) = RF(151) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(27)), (EG(6)*EG(26)));
    RB(152) = RF(152) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(27)), (EG(7)*EG(26)));
    RB(153) = RF(153) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(27)), (EG(5)*EG(14)*EG(17)*PFAC));
    RB(154) = RF(154) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(22), (EG(1)*EG(20)*PFAC));
    RB(155) = RF(155) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(22)*PFAC), EG(23));
    RB(156) = RF(156) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(22)), (EG(1)*EG(21)));
    RB(157) = RF(157) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(22)), (EG(5)*EG(21)));
    RB(158) = RF(158) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(22)), (EG(12)*EG(16)));
    RB(159) = RF(159) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(22)), (EG(10)*EG(17)));
    RB(160) = RF(160) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(22)), (EG(6)*EG(21)));
    RB(161) = RF(161) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(22)), (EG(7)*EG(21)));
    RB(162) = RF(162) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(22)), (EG(5)*EG(28)));
    RB(163) = RF(163) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(22)), (EG(14)*EG(23)));
    RB(164) = RF(164) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(22)), (EG(2)*EG(29)));
    RB(165) = RF(165) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(22)), (EG(13)*EG(20)));
    RB(166) = RF(166) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(22)), (EG(2)*EG(29)));
    RB(167) = RF(167) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(22)), (EG(13)*EG(21)));
    RB(168) = RF(168) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(22)*PFAC), EG(31));
    RB(169) = RF(169) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv =  DIV ((EG(2)*EG(23)*PFAC), EG(24));
    RB(170) = RF(170) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(23)), (EG(1)*EG(22)));
    RB(171) = RF(171) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(23)), (EG(12)*EG(17)));
    RB(172) = RF(172) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(23)), (EG(2)*EG(28)));
    RB(173) = RF(173) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(23)), (EG(7)*EG(22)));
    RB(174) = RF(174) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(23)), (EG(4)*EG(24)));
    RB(175) = RF(175) * MIN(rtemp_inv, SMALL_INV);
}
