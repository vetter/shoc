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
ratx_kernel(__global const real* T, __global const real* C,
		__global real* RF, __global real* RB, __global const real* RKLOW,
		const real TCONV)
{
    const real TEMP = T[get_global_id(0)]*TCONV;
    const real ALOGT = LOG((TEMP));
    real CTOT = 0.0;
    real PR, PCOR, PRLOG, FCENT, FCLOG, XN;
    real CPRLOG, FLOG, FC;
    real SQR;
#ifdef DOUBLE_PRECISION
    const real SMALL = CPREC(1.0e-200);
#else
    const real SMALL = FLT_MIN;
#endif

//#pragma unroll 22
    for (unsigned int k=1; k<=22; k++) {
        CTOT += C(k);
    }

    real CTB_10 = CTOT + C(1) + 5.e0*C(6) + C(10) + 5.e-1*C(11) + C(12)
                + 2.e0*C(16) + 2.e0*C(14) + 2.e0*C(15);
    real CTB_114= CTOT + C(1) + 5.e0*C(6) + C(10) + 5.e-1*C(11) + C(12)
                + 2.e0*C(16) + CPREC(1.5e0)*C(14) + CPREC(1.5e0)*C(15) ;
    real CTB_16 = CTOT + C(1) + 5.e0*C(6) + C(10) + 5.e-1*C(11) + C(12)
                + 2.e0*C(16) + 2.e0*C(14) + 2.e0*C(15) ;

    //     If fall-off (pressure correction):

    PR = RKLOW(1) * DIV(CTB_16, RF(16));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(2.654e-1)*EXP(DIV(-TEMP,CPREC(9.4e1))) + CPREC(7.346e-1)*EXP(DIV(-TEMP,CPREC(1.756e3)))
    + EXP(DIV(-CPREC(5.182e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(16) = RF(16) * PCOR;
    RB(16) = RB(16) * PCOR;

    PR = RKLOW(2) * DIV(CTB_10, RF(31));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(6.8e-2)*EXP(DIV(-TEMP,CPREC(1.97e2))) + CPREC(9.32e-1)*EXP(DIV(-TEMP,CPREC(1.54e3)))
    + EXP(DIV(-CPREC(1.03e4),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV (CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV (FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(31) = RF(31) * PCOR;
    RB(31) = RB(31) * PCOR;

    PR = RKLOW(3) * DIV(CTB_10, RF(39));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(4.243e-1)*EXP(DIV(-TEMP,CPREC(2.37e2))) + CPREC(5.757e-1)*EXP(DIV(-TEMP,CPREC(1.652e3)))
    + EXP(DIV(-CPREC(5.069e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(39) = RF(39) * PCOR;
    RB(39) = RB(39) * PCOR;

    PR = RKLOW(4) * DIV(CTB_10, RF(41));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(2.176e-1)*EXP(DIV(-TEMP,CPREC(2.71e2))) + CPREC(7.824e-1)*EXP(DIV(-TEMP,CPREC(2.755e3)))
    + EXP(DIV(-CPREC(6.57e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(41) = RF(41) * PCOR;
    RB(41) = RB(41) * PCOR;

    PR = RKLOW(5) * DIV(CTB_10, RF(48));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(3.2e-1)*EXP(DIV(-TEMP,CPREC(7.8e1))) + CPREC(6.8e-1)*EXP(DIV(-TEMP,CPREC(1.995e3)))
    + EXP(DIV(-CPREC(5.59e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(48) = RF(48) * PCOR;
    RB(48) = RB(48) * PCOR;

    PR = RKLOW(6) * DIV(CTB_10, RF(56));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(4.093e-1)*EXP(DIV(-TEMP,CPREC(2.75e2))) + CPREC(5.907e-1)*EXP(DIV(-TEMP,CPREC(1.226e3)))
    + EXP(DIV(-CPREC(5.185e3), TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(56) = RF(56) * PCOR;
    RB(56) = RB(56) * PCOR;

    PR = RKLOW(7) * DIV(CTB_10, RF(71));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(2.42e-1)*EXP(DIV(-TEMP,CPREC(9.4e1))) + CPREC(7.58e-1)*EXP(DIV(-TEMP,CPREC(1.555e3)))
    + EXP(DIV(-CPREC(4.2e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(71) = RF(71) * PCOR;
    RB(71) = RB(71) * PCOR;

    PR = RKLOW(8) * DIV(CTB_10, RF(78));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(2.17e-1)*EXP(DIV(-TEMP,CPREC(7.4e1))) + CPREC(7.83e-1)*EXP(DIV(-TEMP,CPREC(2.941e3)))
    + EXP(DIV(-CPREC(6.964e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(78) = RF(78) * PCOR;
    RB(78) = RB(78) * PCOR;

    PR = RKLOW(9) * DIV(CTB_10, RF(89));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(3.827e-1)*EXP(DIV(-TEMP,CPREC(1.3076e1))) + CPREC(6.173e-1)*EXP(DIV(-TEMP,CPREC(2.078e3)))
    + EXP(DIV(-CPREC(5.093e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(89) = RF(89) * PCOR;
    RB(89) = RB(89) * PCOR;

    PR = RKLOW(10) * DIV(CTB_10, RF(93));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(4.675e-1)*EXP(DIV(-TEMP,CPREC(1.51e2))) + CPREC(5.325e-1)*EXP(DIV(-TEMP,CPREC(1.038e3)))
    + EXP(DIV(-CPREC(4.97e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(93) = RF(93) * PCOR;
    RB(93) = RB(93) * PCOR;

    PR = RKLOW(11) * DIV(CTB_114, RF(114));
    PCOR = DIV(PR, (1.0 + PR));
    RF(114) = RF(114) * PCOR;
    RB(114) = RB(114) * PCOR;

    PR = RKLOW(12) * DIV(CTB_10, RF(115));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = -CPREC(9.816e-1)*EXP(DIV(-TEMP,CPREC(5.3837e3))) +
    		CPREC(1.9816e0)*EXP(DIV(-TEMP,CPREC(4.2932e0)))  +
    		EXP(DIV(CPREC(7.95e-2),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(115) = RF(115) * PCOR;
    RB(115) = RB(115) * PCOR;
}
