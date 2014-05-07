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
ratt3_kernel(__global const real* T, __global const real* RF,
		__global real* RB, __global	const real* EG, const real TCONV)
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

    rtemp_inv = DIV ((EG(3)*EG(8)), (EG(5)*EG(7)));
    RB(26) = RF(26) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(8)), (EG(6)*EG(7)));
    RB(27) = RF(27) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(8)), (EG(6)*EG(7)));
    RB(28) = RF(28) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(14)*PFAC), EG(15));
    RB(29) = RF(29) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(14)), (EG(2)*EG(15)));
    RB(30) = RF(30) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(14)*PFAC), EG(17));
    RB(31) = RF(31) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(14)), (EG(3)*EG(15)));
    RB(32) = RF(32) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(14)), (EG(5)*EG(15)));
    RB(33) = RF(33) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(9)), (EG(2)*EG(14)));
    RB(34) = RF(34) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(9)), (EG(2)*EG(16)));
    RB(35) = RF(35) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(9)), (EG(2)*EG(10)));
    RB(36) = RF(36) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(6)*EG(9)), (EG(2)*EG(17)));
    RB(37) = RF(37) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(9)), (EG(3)*EG(16)));
    RB(38) = RF(38) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(14)*PFAC), EG(25));
    RB(39) = RF(39) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(15)), (EG(14)*EG(16)));
    RB(40) = RF(40) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(16)*PFAC), EG(17));
    RB(41) = RF(41) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(16)), (EG(1)*EG(14)));
    RB(42) = RF(42) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(16)), (EG(5)*EG(14)));
    RB(43) = RF(43) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(16)), (EG(2)*EG(15)));
    RB(44) = RF(44) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(16)), (EG(6)*EG(14)));
    RB(45) = RF(45) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*PFAC), (EG(2)*EG(14)));
    RB(46) = RF(46) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(16)), (EG(7)*EG(14)));
    RB(47) = RF(47) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(10)*PFAC), EG(12));
    RB(48) = RF(48) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(10)), (EG(2)*EG(12)));
    RB(49) = RF(49) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(10)), (EG(2)*EG(16)));
    RB(50) = RF(50) * MIN(rtemp_inv, SMALL_INV);

}
