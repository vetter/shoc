#include <stdio.h>
#include <stdlib.h>

#include "CTimer.h"

void 
DoTriadFloats( unsigned int nItems, 
                    unsigned int blockSize,
                    const float* A, 
                    const float* B, 
                    float s, 
                    float* C, 
                    double* TriadTime)
{
    const float *restrict a[2];
    const float *restrict b[2];
    float *restrict c[2];

    int i;

    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    for( int j=0; j < nItems/blockSize/2; j++)
    {
        // get my chunk of the array to work on
        a[0] = &A[2*j*blockSize]; 
        b[0] = &B[2*j*blockSize]; 
        c[0] = &C[2*j*blockSize]; 
        a[1] = &A[(2*j+1)*blockSize]; 
        b[1] = &B[(2*j+1)*blockSize]; 
        c[1] = &C[(2*j+1)*blockSize]; 
        #pragma acc data create(a[0:2][0:blockSize],b[0:2][0:blockSize],c[0:2][0:blockSize])
        {
            //first stream
            #pragma acc update device(a[0][0:blockSize],b[0][0:blockSize]) async(1)
            #pragma acc kernels async(1)
            for( i=0; i < blockSize; i++ )
            {
                c[0][i] = a[0][i] + s*b[0][i];
            }
            #pragma acc update host(c[0][0:blockSize]) async(1)
            #pragma acc wait(1)

            //second stream
            #pragma acc update device(a[1][0:blockSize],b[1][0:blockSize]) async(2)
            #pragma acc kernels async(2)
            for( i=0; i < blockSize; i++ ) 
            {
                c[1][i] = a[1][i] + s*b[1][i];
            }
            #pragma acc update host(c[1][0:blockSize]) async(2)
            #pragma acc wait(2)
        }
    }
    printf("C[5] = %f | C[-5] = %f\n",C[5],C[nItems-5]);

    // stop the timer and record the result (in seconds)
    *TriadTime = Timer_Stop( wholeTimerHandle, "" );

}

void 
DoTriadDoubles( unsigned int nItems, 
                     unsigned int blockSize,
                     double* A, 
                     double* B, 
                     double s, 
                     double* C, 
                     double* TriadTime)
{
    int n = nItems;
    const double *restrict a[2];
    const double *restrict b[2];
    double *restrict c[2];

    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    for( int j=0; j < nItems/blockSize/2; j++)
    {
        // get my chunk of the array to work on
        a[0] = &A[2*j*blockSize]; 
        b[0] = &B[2*j*blockSize]; 
        c[0] = &C[2*j*blockSize]; 
        a[1] = &A[(2*j+1)*blockSize]; 
        b[1] = &B[(2*j+1)*blockSize]; 
        c[1] = &C[(2*j+1)*blockSize]; 
        #pragma acc data create(a[0:2][0:blockSize],b[0:2][0:blockSize],c[0:2][0:blockSize])
        {
            //first stream
            #pragma acc update device(a[0][0:blockSize],b[0][0:blockSize]) async(1)
            #pragma acc kernels async(1)
            for( int i=0; i < blockSize; i++ )
            {
                c[0][i] = a[0][i] + s*b[0][i];
            }
            #pragma acc update host(c[0][0:blockSize]) async(1)
            #pragma acc wait(1)

            //second stream
            #pragma acc update device(a[1][0:blockSize],b[1][0:blockSize]) async(2)
            #pragma acc kernels async(2)
            for( int i=0; i < blockSize; i++ ) 
            {
                c[1][i] = a[1][i] + s*b[1][i];
            }
            #pragma acc update host(c[1][0:blockSize]) async(2)
            #pragma acc wait(2)
        }
    }

    // stop the timer and record the result (in seconds)
    *TriadTime = Timer_Stop( wholeTimerHandle, "" );

}

