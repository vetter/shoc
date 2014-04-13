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
ratt2_kernel(__global const real* T, __global const real* RF, __global real* RB,
		__global const real* EG, const real TCONV)
{

    const real TEMP = T[get_global_id(0)]*TCONV;
    const real ALOGT = LOG(TEMP);
#ifdef DOUBLE_PRECISION
    const real SMALL_INV = 1e300;
#else
    const real SMALL_INV = 1e+20f;
#endif
    const real RU=CPREC(8.31451e7);
    const real PATM = CPREC(1.01325e6);
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    real rtemp_inv;

    rtemp_inv = DIV ((EG(2)*EG(4)), (EG(3)*EG(5)));
    RB(1) = RF(1) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(3)), (EG(2)*EG(5)));
    RB(2) = RF(2) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(5)), (EG(2)*EG(6)));
    RB(3) = RF(3) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(5)), (EG(3)*EG(6)));
    RB(4) = RF(4) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(5) = RF(5) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(6) = RF(6) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(7) = RF(7) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(8) = RF(8) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(5)*PFAC), EG(6));
    RB(9) = RF(9) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(3)*PFAC), EG(5));
    RB(10) = RF(10) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(3)*PFAC), EG(4));
    RB(11) = RF(11) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(12) = RF(12) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(13) = RF(13) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(14) = RF(14) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(15) = RF(15) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(5)*PFAC), EG(8));
    RB(16) = RF(16) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(3)*EG(6)));
    RB(17) = RF(17) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(1)*EG(4)));
    RB(18) = RF(18) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(5)*EG(5)));
    RB(19) = RF(19) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(7)), (EG(4)*EG(5)));
    RB(20) = RF(20) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(7)), (EG(4)*EG(6)));
    RB(21) = RF(21) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(7)), (EG(4)*EG(8)));
    RB(22) = RF(22) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(7)), (EG(4)*EG(8)));
    RB(23) = RF(23) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(8)), (EG(1)*EG(7)));
    RB(24) = RF(24) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(8)), (EG(5)*EG(6)));
    RB(25) = RF(25) * MIN(rtemp_inv, SMALL_INV);
}
