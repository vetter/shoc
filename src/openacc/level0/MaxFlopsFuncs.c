#include <stdio.h>
#include <stdlib.h>
#include "CTimer.h"

// v = 10.0
#define ADD1_OP   s=v-s;
#define ADD2_OP   ADD1_OP s2=v-s2;
#define ADD4_OP   ADD2_OP s3=v-s3; s4=v-s4;
#define ADD8_OP   ADD4_OP s5=v-s5; s6=v-s6; s7=v-s7; s8=v-s8;

// v = 1.01
#define MUL1_OP   s=s*s*v;
#define MUL2_OP   MUL1_OP s2=s2*s2*v;
#define MUL4_OP   MUL2_OP s3=s3*s3*v; s4=s4*s4*v;
#define MUL8_OP   MUL4_OP s5=s5*s5*v; s6=s6*s6*v; s7=s7*s7*v; s8=s8*s8*v;

// v1 = 10.0, v2 = 0.9899
#define MADD1_OP  s=v-s*v2;
#define MADD2_OP  MADD1_OP s2=v-s2*v2;
#define MADD4_OP  MADD2_OP s3=v-s3*v2; s4=v-s4*v2;
#define MADD8_OP  MADD4_OP s5=v-s5*v2; s6=v-s6*v2; s7=v-s7*v2; s8=v-s8*v2;

// v1 = 3.75, v2 = 0.355
#define MULMADD1_OP  s=(v-v2*s)*s;
#define MULMADD2_OP  MULMADD1_OP s2=(v-v2*s2)*s2;
#define MULMADD4_OP  MULMADD2_OP s3=(v-v2*s3)*s3; s4=(v-v2*s4)*s4;
#define MULMADD8_OP  MULMADD4_OP s5=(v-v2*s5)*s5; s6=(v-v2*s6)*s6; s7=(v-v2*s7)*s7; s8=(v-v2*s8)*s8;

#define ADD1_MOP20  \
     ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP \
     ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP
#define ADD2_MOP20  \
     ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP \
     ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP
#define ADD4_MOP10  \
     ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP \
     ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP
#define ADD8_MOP5  \
     ADD8_OP ADD8_OP ADD8_OP ADD8_OP ADD8_OP

#define MUL1_MOP20  \
     MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP \
     MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP
#define MUL2_MOP20  \
     MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP \
     MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP
#define MUL4_MOP10  \
     MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP \
     MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP
#define MUL8_MOP5  \
     MUL8_OP MUL8_OP MUL8_OP MUL8_OP MUL8_OP

#define MADD1_MOP20  \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP
#define MADD2_MOP20  \
     MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP \
     MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP
#define MADD4_MOP10  \
     MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP \
     MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP
#define MADD8_MOP5  \
     MADD8_OP MADD8_OP MADD8_OP MADD8_OP MADD8_OP

#define MULMADD1_MOP20  \
     MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP \
     MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP
#define MULMADD2_MOP20  \
     MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP \
     MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP
#define MULMADD4_MOP10  \
     MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP \
     MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP
#define MULMADD8_MOP5  \
     MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP

void Add1Float(float *data, int numFloats, int nIters, float v, double *kTime, double *tTime)
{
    int j,i;
    float s;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 12 more times for 240 operations total.
                 */
                s = data[i];
                ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 
                ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 
                data[i] = s;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Add1Double(double *data, int numFloats, int nIters, double v, double *kTime, double *tTime)
{
    int j,i;
    double s;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 12 more times for 240 operations total.
                 */
                s = data[i];
                ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 
                ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 
                data[i] = s;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Add2Float(float *data, int numFloats, int nIters, float v, double *kTime, double *tTime)
{
    int j,i;
    float s, s2;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 6 more times for 120 operations total.
                */
                s = data[i]; s2=10.0f-s;
                ADD2_MOP20 ADD2_MOP20 ADD2_MOP20
                ADD2_MOP20 ADD2_MOP20 ADD2_MOP20
                data[i] = s+s2;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Add2Double(double *data, int numFloats, int nIters, double v, double *kTime, double *tTime)
{
    int j,i;
    double s,s2;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 6 more times for 120 operations total.
                */
                s = data[i]; s2=10.0f-s;
                ADD2_MOP20 ADD2_MOP20 ADD2_MOP20
                ADD2_MOP20 ADD2_MOP20 ADD2_MOP20
                data[i] = s+s2;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Add4Float(float *data, int numFloats, int nIters, float v, double *kTime, double *tTime)
{
    int j,i;
    float s, s2, s3, s4;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 10 operations. 
                   Unroll 6 more times for 60 operations total.
                 */
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2;
                ADD4_MOP10 ADD4_MOP10 ADD4_MOP10
                ADD4_MOP10 ADD4_MOP10 ADD4_MOP10
                data[i] = (s+s2)+(s3+s4);
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Add4Double(double *data, int numFloats, int nIters, double v, double *kTime, double *tTime)
{
    int j,i;
    double s, s2, s3, s4;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 10 operations. 
                   Unroll 6 more times for 60 operations total.
                 */
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2;
                ADD4_MOP10 ADD4_MOP10 ADD4_MOP10
                ADD4_MOP10 ADD4_MOP10 ADD4_MOP10
                data[i] = (s+s2)+(s3+s4);
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Add8Float(float *data, int numFloats, int nIters, float v, double *kTime, double *tTime)
{
    int j,i;
    float s, s2, s3, s4, s5, s6, s7, s8;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 5 operations. 
                   Unroll 6 more times for 30 operations total.
                 */
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2; 
                s5=8.0f-s; s6=8.0f-s2; s7=7.0f-s; s8=7.0f-s2;
                ADD8_MOP5 ADD8_MOP5 ADD8_MOP5
                ADD8_MOP5 ADD8_MOP5 ADD8_MOP5
                data[i] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Add8Double(double *data, int numFloats, int nIters, double v, double *kTime, double *tTime)
{
    int j,i;
    double s, s2, s3, s4, s5, s6, s7, s8;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 5 operations. 
                   Unroll 6 more times for 30 operations total.
                 */
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2; 
                s5=8.0f-s; s6=8.0f-s2; s7=7.0f-s; s8=7.0f-s2;
                ADD8_MOP5 ADD8_MOP5 ADD8_MOP5
                ADD8_MOP5 ADD8_MOP5 ADD8_MOP5
                data[i] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Mul1Float(float *data, int numFloats, int nIters, float v, double *kTime, double *tTime)
{
    int j,i;
    float s;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 10 more times for 200 operations total.
                */
                //s = data[i]-data[i]+0.999f; //FIXME why doesn't this work?
                s = data[i];
                MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
                MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
                data[i] = s;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Mul1Double(double *data, int numFloats, int nIters, double v, double *kTime, double *tTime)
{
    int j,i;
    double s;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 10 more times for 200 operations total.
                */
                //s = data[i]-data[i]+0.999f; //FIXME why doesn't this work?
                s = data[i];
                MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
                MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
                data[i] = s;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Mul2Float(float *data, int numFloats, int nIters, float v, double *kTime, double *tTime)
{
    int j,i;
    float s, s2;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 5 more times for 100 operations total.
                */
                //s = data[i]-data[i]+0.999f; s2=s-0.0001f;
                s = data[i]; s2=s-0.0001f;
                MUL2_MOP20 MUL2_MOP20 MUL2_MOP20
                MUL2_MOP20 MUL2_MOP20
                data[i] = s+s2;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Mul2Double(double *data, int numFloats, int nIters, double v, double *kTime, double *tTime)
{
    int j,i;
    double s,s2;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 5 more times for 100 operations total.
                */
                //s = data[i]-data[i]+0.999f; s2=s-0.0001f;
                s = data[i]; s2=s-0.0001f;
                MUL2_MOP20 MUL2_MOP20 MUL2_MOP20
                MUL2_MOP20 MUL2_MOP20
                data[i] = s+s2;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Mul4Float(float *data, int numFloats, int nIters, float v, double *kTime, double *tTime)
{
    int j,i;
    float s, s2, s3, s4;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 10 operations. 
                   Unroll 5 more times for 50 operations total.
                */
                //s = data[i]-data[i]+0.999f; s2=s-0.0001f; s3=s-0.0002f; s4=s-0.0003f;
                s = data[i]; s2=s-0.0001f; s3=s-0.0002f; s4=s-0.0003f;
                MUL4_MOP10 MUL4_MOP10 MUL4_MOP10
                MUL4_MOP10 MUL4_MOP10
                data[i] = (s+s2)+(s3+s4);
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Mul4Double(double *data, int numFloats, int nIters, double v, double *kTime, double *tTime)
{
    int j,i;
    double s, s2, s3, s4;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 10 operations. 
                   Unroll 5 more times for 50 operations total.
                */
                //s = data[i]-data[i]+0.999f; s2=s-0.0001f; s3=s-0.0002f; s4=s-0.0003f;
                s = data[i]; s2=s-0.0001f; s3=s-0.0002f; s4=s-0.0003f;
                MUL4_MOP10 MUL4_MOP10 MUL4_MOP10
                MUL4_MOP10 MUL4_MOP10
                data[i] = (s+s2)+(s3+s4);
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Mul8Float(float *data, int numFloats, int nIters, float v, double *kTime, double *tTime)
{
    int j,i;
    float s, s2, s3, s4, s5, s6, s7, s8;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 5 operations. 
                   Unroll 5 more times for 25 operations total.
                */
                //s = data[i]-data[i]+0.999f; s2=s-0.0001f; s3=s-0.0002f; s4=s-0.0003f; 
                s = data[i]; s2=s-0.0001f; s3=s-0.0002f; s4=s-0.0003f; 
                s5=s-0.0004f; s6=s-0.0005f; s7=s-0.0006f; s8=s-0.0007f;
                MUL8_MOP5 MUL8_MOP5 MUL8_MOP5
                MUL8_MOP5 MUL8_MOP5
                data[i] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void Mul8Double(double *data, int numFloats, int nIters, double v, double *kTime, double *tTime)
{
    int j,i;
    double s, s2, s3, s4, s5, s6, s7, s8;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 5 operations. 
                   Unroll 5 more times for 25 operations total.
                */
                //s = data[i]-data[i]+0.999f; s2=s-0.0001f; s3=s-0.0002f; s4=s-0.0003f; 
                s = data[i]; s2=s-0.0001f; s3=s-0.0002f; s4=s-0.0003f; 
                s5=s-0.0004f; s6=s-0.0005f; s7=s-0.0006f; s8=s-0.0007f;
                MUL8_MOP5 MUL8_MOP5 MUL8_MOP5
                MUL8_MOP5 MUL8_MOP5
                data[i] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MAdd1Float(float *data, int numFloats, int nIters, float v, float v2, double *kTime, double *tTime)
{
    int j,i;
    float s;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 12 more times for 240 operations total.
                */
                s = data[i];
                MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 
                MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 
                data[i] = s;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MAdd1Double(double *data, int numFloats, int nIters, double v, double v2, double *kTime, double *tTime)
{
    int j,i;
    double s;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 12 more times for 240 operations total.
                */
                s = data[i];
                MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 
                MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 
                data[i] = s;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MAdd2Float(float *data, int numFloats, int nIters, float v, float v2, double *kTime, double *tTime)
{
    int j,i;
    float s, s2;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 6 more times for 120 operations total.
                */
                s = data[i]; s2=10.0f-s;
                MADD2_MOP20 MADD2_MOP20 MADD2_MOP20
                MADD2_MOP20 MADD2_MOP20 MADD2_MOP20
                data[i] = s+s2;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MAdd2Double(double *data, int numFloats, int nIters, double v, double v2, double *kTime, double *tTime)
{
    int j,i;
    double s,s2;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                /* Each macro op has 20 operations. 
                   Unroll 6 more times for 120 operations total.
                */
                s = data[i]; s2=10.0f-s;
                MADD2_MOP20 MADD2_MOP20 MADD2_MOP20
                MADD2_MOP20 MADD2_MOP20 MADD2_MOP20
                data[i] = s+s2;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MAdd4Float(float *data, int numFloats, int nIters, float v, float v2, double *kTime, double *tTime)
{
    int j,i;
    float s, s2, s3, s4;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2;
                /* Each macro op has 10 operations. 
                   Unroll 6 more times for 60 operations total.
                */
                MADD4_MOP10 MADD4_MOP10 MADD4_MOP10
                MADD4_MOP10 MADD4_MOP10 MADD4_MOP10
                data[i] = (s+s2)+(s3+s4);
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MAdd4Double(double *data, int numFloats, int nIters, double v, double v2, double *kTime, double *tTime)
{
    int j,i;
    double s, s2, s3, s4;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2;
                /* Each macro op has 10 operations. 
                   Unroll 6 more times for 60 operations total.
                */
                MADD4_MOP10 MADD4_MOP10 MADD4_MOP10
                MADD4_MOP10 MADD4_MOP10 MADD4_MOP10
                data[i] = (s+s2)+(s3+s4);
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MAdd8Float(float *data, int numFloats, int nIters, float v, float v2, double *kTime, double *tTime)
{
    int j,i;
    float s, s2, s3, s4, s5, s6, s7, s8;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2; 
                s5=8.0f-s; s6=8.0f-s2; s7=7.0f-s; s8=7.0f-s2;
                /* Each macro op has 5 operations. 
                   Unroll 6 more times for 30 operations total.
                */
                MADD8_MOP5 MADD8_MOP5 MADD8_MOP5
                MADD8_MOP5 MADD8_MOP5 MADD8_MOP5
                data[i] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MAdd8Double(double *data, int numFloats, int nIters, double v, double v2, double *kTime, double *tTime)
{
    int j,i;
    double s, s2, s3, s4, s5, s6, s7, s8;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2; 
                s5=8.0f-s; s6=8.0f-s2; s7=7.0f-s; s8=7.0f-s2;
                /* Each macro op has 5 operations. 
                   Unroll 6 more times for 30 operations total.
                */
                MADD8_MOP5 MADD8_MOP5 MADD8_MOP5
                MADD8_MOP5 MADD8_MOP5 MADD8_MOP5
                data[i] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MulMAdd1Float(float *data, int numFloats, int nIters, float v, float v2, double *kTime, double *tTime)
{
    int j,i;
    float s;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i];
                /* Each macro op has 20 operations. 
                   Unroll 8 more times for 160 operations total.
                */
                MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
                MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
                data[i] = s;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MulMAdd1Double(double *data, int numFloats, int nIters, double v, double v2, double *kTime, double *tTime)
{
    int j,i;
    double s;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i];
                /* Each macro op has 20 operations. 
                   Unroll 8 more times for 160 operations total.
                */
                MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
                MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
                data[i] = s;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MulMAdd2Float(float *data, int numFloats, int nIters, float v, float v2, double *kTime, double *tTime)
{
    int j,i;
    float s, s2;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i], s2=10.0f-s;
                /* Each macro op has 20 operations. 
                   Unroll 4 more times for 80 operations total.
                */
                MULMADD2_MOP20 MULMADD2_MOP20
                MULMADD2_MOP20 MULMADD2_MOP20
                data[i] = s+s2;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MulMAdd2Double(double *data, int numFloats, int nIters, double v, double v2, double *kTime, double *tTime)
{
    int j,i;
    double s,s2;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i], s2=10.0f-s;
                /* Each macro op has 20 operations. 
                   Unroll 4 more times for 80 operations total.
                */
                MULMADD2_MOP20 MULMADD2_MOP20
                MULMADD2_MOP20 MULMADD2_MOP20
                data[i] = s+s2;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MulMAdd4Float(float *data, int numFloats, int nIters, float v, float v2, double *kTime, double *tTime)
{
    int j,i;
    float s, s2, s3, s4;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2;
                /* Each macro op has 10 operations. 
                   Unroll 4 more times for 40 operations total.
                */
                MULMADD4_MOP10 MULMADD4_MOP10
                MULMADD4_MOP10 MULMADD4_MOP10
                data[i] = (s+s2)+(s3+s4);
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MulMAdd4Double(double *data, int numFloats, int nIters, double v, double v2, double *kTime, double *tTime)
{
    int j,i;
    double s, s2, s3, s4;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2;
                /* Each macro op has 10 operations. 
                   Unroll 4 more times for 40 operations total.
                */
                MULMADD4_MOP10 MULMADD4_MOP10
                MULMADD4_MOP10 MULMADD4_MOP10
                data[i] = (s+s2)+(s3+s4);
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MulMAdd8Float(float *data, int numFloats, int nIters, float v, float v2, double *kTime, double *tTime)
{
    int j,i;
    float s, s2, s3, s4, s5, s6, s7, s8;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2; 
                s5=8.0f-s; s6=8.0f-s2; s7=7.0f-s; s8=7.0f-s2;
                /* Each macro op has 5 operations. 
                   Unroll 4 more times for 20 operations total.
                */
                MULMADD8_MOP5 MULMADD8_MOP5
                MULMADD8_MOP5 MULMADD8_MOP5
                data[i] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

void MulMAdd8Double(double *data, int numFloats, int nIters, double v, double v2, double *kTime, double *tTime)
{
    int j,i;
    double s, s2, s3, s4, s5, s6, s7, s8;

    int wholeTimerHandle = Timer_Start();
    #pragma acc data copy(data[0:numFloats])
    {
        int iterTimerHandle = Timer_Start();

        #pragma acc kernels
        for (j=0 ; j<nIters ; ++j) {
            for (i=0 ; i<numFloats; ++i) {
                s = data[i]; s2=10.0f-s; s3=9.0f-s; s4=9.0f-s2; 
                s5=8.0f-s; s6=8.0f-s2; s7=7.0f-s; s8=7.0f-s2;
                /* Each macro op has 5 operations. 
                   Unroll 4 more times for 20 operations total.
                */
                MULMADD8_MOP5 MULMADD8_MOP5
                MULMADD8_MOP5 MULMADD8_MOP5
                data[i] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;

}

