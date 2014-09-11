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
ratt7_kernel(__global const real* T, __global const real* RF,
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

    rtemp_inv = DIV ((EG(2)*EG(26)*PFAC), EG(27));
    RB(126) = RF(126) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(26)), (EG(1)*EG(25)));
    RB(127) = RF(127) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(26)), (EG(12)*EG(14)));
    RB(128) = RF(128) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(26)), (EG(5)*EG(25)));
    RB(129) = RF(129) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(26)), (EG(10)*EG(15)));
    RB(130) = RF(130) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(26)), (EG(6)*EG(25)));
    RB(131) = RF(131) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)*PFAC), EG(22));
    RB(132) = RF(132) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)), (EG(1)*EG(19)));
    RB(133) = RF(133) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)), (EG(1)*EG(20)));
    RB(134) = RF(134) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(21)), (EG(2)*EG(26)));
    RB(135) = RF(135) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(21)), (EG(12)*EG(14)));
    RB(136) = RF(136) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(21)), (EG(6)*EG(19)));
    RB(137) = RF(137) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(7)*EG(19)));
    RB(138) = RF(138) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(3)*EG(27)));
    RB(139) = RF(139) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(16)*EG(17)));
    RB(140) = RF(140) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(21)), (EG(5)*EG(27)));
    RB(141) = RF(141) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(8)*EG(21)), (EG(7)*EG(22)));
    RB(142) = RF(142) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(21)), (EG(14)*EG(22)));
    RB(143) = RF(143) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)), (EG(13)*EG(19)));
    RB(144) = RF(144) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)*PFAC), EG(30));
    RB(145) = RF(145) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)), (EG(2)*EG(29)));
    RB(146) = RF(146) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(27), (EG(12)*EG(14)*PFAC));
    RB(147) = RF(147) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)*PFAC), EG(28));
    RB(148) = RF(148) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)), (EG(12)*EG(16)));
    RB(149) = RF(149) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)), (EG(1)*EG(26)));
    RB(150) = RF(150) * MIN(rtemp_inv, SMALL_INV);
}
