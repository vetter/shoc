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
ratxb_kernel(__global const real* T, __global const real* C, __global real* RF,
		__global real* RB, __global const real* RKLOW, const real TCONV)
{

    const real TEMP = T[get_global_id(0)]*TCONV;
    const real ALOGT = LOG((TEMP));
    real CTOT = 0.0f;
    real PR, PCOR, PRLOG, FCENT, FCLOG, XN;
    real CPRLOG, FLOG, FC, SQR;

#ifdef DOUBLE_PRECISION
    const real SMALL = CPREC(1.0e-200);
#else
    const real SMALL = FLT_MIN;
#endif

//#pragma unroll 22
    for (unsigned int k=1; k<=22; k++)
    {
        CTOT += C(k);
    }

    real CTB_5  = CTOT - C(1) - C(6) + C(10) - C(12) + 2.e0*C(16)
                + 2.e0*C(14) + 2.e0*C(15) ;
    real CTB_9  = CTOT - CPREC(2.7e-1)*C(1) + CPREC(2.65e0)*C(6) + C(10) + 2.e0*C(16)
                + 2.e0*C(14) + 2.e0*C(15) ;
    real CTB_10 = CTOT + C(1) + 5.e0*C(6) + C(10) + 5.e-1*C(11) + C(12)
                + 2.e0*C(16) + 2.e0*C(14) + 2.e0*C(15);
    real CTB_11 = CTOT + CPREC(1.4e0)*C(1) + CPREC(1.44e1)*C(6) + C(10) + CPREC(7.5e-1)*C(11)
                + CPREC(2.6e0)*C(12) + 2.e0*C(16) + 2.e0*C(14)
                + 2.e0*C(15) ;
    real CTB_12 = CTOT - C(4) - C(6) - CPREC(2.5e-1)*C(11) + 5.e-1*C(12)
                + 5.e-1*C(16) - C(22) + 2.e0*C(14) + 2.e0*C(15) ;
    real CTB_29 = CTOT + C(1) + 5.e0*C(4) + 5.e0*C(6) + C(10)
                + 5.e-1*C(11) + CPREC(2.5e0)*C(12) + 2.e0*C(16)
                + 2.e0*C(14) + 2.e0*C(15) ;
    real CTB_190= CTOT + C(1) + 5.e0*C(6) + C(10) + 5.e-1*C(11)
                + C(12) + 2.e0*C(16) ;

    RF(5) = RF(5)*CTB_5*C(2)*C(2);
    RB(5) = RB(5)*CTB_5*C(1);
    RF(9) = RF(9)*CTB_9*C(2)*C(5);
    RB(9) = RB(9)*CTB_9*C(6);
    RF(10) = RF(10)*CTB_10*C(3)*C(2);
    RB(10) = RB(10)*CTB_10*C(5);
    RF(11) = RF(11)*CTB_11*C(3)*C(3);
    RB(11) = RB(11)*CTB_11*C(4);
    RF(12) = RF(12)*CTB_12*C(2)*C(4);
    RB(12) = RB(12)*CTB_12*C(7);
    RF(29) = RF(29)*CTB_29*C(11)*C(3);
    RB(29) = RB(29)*CTB_29*C(12);
    RF(46) = RF(46)*CTB_10;
    RB(46) = RB(46)*CTB_10*C(11)*C(2);
    RF(121) = RF(121)*CTOT*C(14)*C(9);
    RB(121) = RB(121)*CTOT*C(20);


    PR = RKLOW(13) * DIV(CTB_10, RF(126));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(6.63e-1)*EXP(DIV(-TEMP,CPREC(1.707e3))) + CPREC(3.37e-1)*EXP(DIV(-TEMP,CPREC(3.2e3)))
    + EXP(DIV(-CPREC(4.131e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(126) = RF(126) * PCOR;
    RB(126) = RB(126) * PCOR;

    PR = RKLOW(14) * DIV(CTB_10, RF(132));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(2.18e-1)*EXP(DIV(-TEMP,CPREC(2.075e2))) + CPREC(7.82e-1)*EXP(DIV(-TEMP,CPREC(2.663e3)))
    + EXP(DIV(-CPREC(6.095e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(132) = RF(132) * PCOR;
    RB(132) = RB(132) * PCOR;

    PR = RKLOW(15) * DIV(CTB_10, RF(145));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(8.25e-1)*EXP(DIV(-TEMP,CPREC(1.3406e3))) + CPREC(1.75e-1)*EXP(DIV(-TEMP,CPREC(6.e4)))
    + EXP(DIV(-CPREC(1.01398e4),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(145) = RF(145) * PCOR;
    RB(145) = RB(145) * PCOR;

    PR = RKLOW(16) * DIV(CTB_10, RF(148));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(4.5e-1)*EXP(DIV(-TEMP,CPREC(8.9e3))) + CPREC(5.5e-1)*EXP(DIV(-TEMP,CPREC(4.35e3)))
    + EXP(DIV(-CPREC(7.244e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(148) = RF(148) * PCOR;
    RB(148) = RB(148) * PCOR;

    PR = RKLOW(17) * DIV(CTB_10, RF(155));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(2.655e-1)*EXP(DIV(-TEMP,CPREC(1.8e2))) + CPREC(7.345e-1)*EXP(DIV(-TEMP,CPREC(1.035e3)))
    + EXP(DIV(-CPREC(5.417e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(155) = RF(155) * PCOR;
    RB(155) = RB(155) * PCOR;

    PR = RKLOW(18) * DIV(CTB_10, RF(156));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(2.47e-2)*EXP(DIV(-TEMP,CPREC(2.1e2))) + CPREC(9.753e-1)*EXP(DIV(-TEMP,CPREC(9.84e2)))
    + EXP(DIV(-CPREC(4.374e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(156) = RF(156) * PCOR;
    RB(156) = RB(156) * PCOR;

    PR = RKLOW(19) * DIV(CTB_10, RF(170));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(1.578e-1)*EXP(DIV(-TEMP,CPREC(1.25e2))) + CPREC(8.422e-1)*EXP(DIV(-TEMP,CPREC(2.219e3)))
    + EXP(DIV(-CPREC(6.882e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(170) = RF(170) * PCOR;
    RB(170) = RB(170) * PCOR;

    PR = RKLOW(20) * DIV(CTB_10, RF(185));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = CPREC(9.8e-1)*EXP(DIV(-TEMP,CPREC(1.0966e3))) + 2.e-2*EXP(DIV(-TEMP,CPREC(1.0966e3)))
    + EXP(DIV(-CPREC(6.8595e3),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(185) = RF(185) * PCOR;
    RB(185) = RB(185) * PCOR;

    PR = RKLOW(21) * DIV(CTB_190, RF(190));
    PCOR = DIV(PR, (1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = 0.e0*EXP(DIV(-TEMP,CPREC(1.e3))) + 1.e0*EXP(DIV(-TEMP,CPREC(1.31e3)))
    + EXP(DIV(-CPREC(4.8097e4),TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = 0.75 - 1.27*FCLOG;
    CPRLOG= PRLOG - (0.4 + 0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-0.14*CPRLOG));
    FLOG = DIV(FCLOG, (1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(190) = RF(190) * PCOR;
    RB(190) = RB(190) * PCOR;

}
