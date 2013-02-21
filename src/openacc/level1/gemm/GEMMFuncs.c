#include <stdio.h>
#include <stdlib.h>

#include "CTimer.h"

void sgemm(char transa, char transb, int m, int n, int k, 
    float alpha, const float *A, int lda, const float *B, int ldb, 
    float beta, float *C, int ldc, double* kTime, double* tTime)
{
    int i, j, l;

    int wholeTimerHandle = Timer_Start();
    
    #pragma acc data copyout(C[0:(n*n)]), copyin(A[0:(n*n)], B[0:(n*n)]) 
    { 
        int iterTimerHandle = Timer_Start();
        
        #pragma acc kernels loop gang, vector(8)
        for (i = 0; i<m; i++)
        {
            #pragma acc loop gang, vector(8)
            for (j=0; j<n; j++)
            { 
                double sum=0.0;
                #pragma acc loop seq
                for (l=0; l<k; l++)
                {
                    sum+=A[i*n+l]*B[k*n+j];
                }
                C[i*n+j]=sum;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;
}

void dgemm(char transa, char transb, int m, int n, int k, 
    double alpha, const double *A, int lda, const double *B, int ldb, 
    double beta, double *C, int ldc, double* kTime, double* tTime)
{
    int i, j, l;

    int wholeTimerHandle = Timer_Start();
    
    #pragma acc data copyout(C[0:(n*n)]), copyin(A[0:(n*n)], B[0:(n*n)]) 
    { 
        int iterTimerHandle = Timer_Start();
        
        #pragma acc kernels loop gang, vector(8)
        for (i = 0; i<m; i++)
        {
            #pragma acc loop gang, vector(8)
            for (j=0; j<n; j++)
            { 
                double sum=0.0;
                #pragma acc loop seq
                for (l=0; l<k; l++)
                {
                    sum+=A[i*n+l]*B[k*n+j];
                }
                C[i*n+j]=sum;
            }
        }
        *kTime = Timer_Stop( iterTimerHandle, "" );
    }
    *tTime = Timer_Stop( wholeTimerHandle, "" ) - *kTime;
}

