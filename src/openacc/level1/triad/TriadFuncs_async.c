#include <stdio.h>
#include <stdlib.h>

#include "CTimer.h"

void 
DoTriadFloatsIters( unsigned int nIters, 
                    unsigned int nItems, 
                    const float* A, 
                    const float* B, 
                    float s, 
                    float* C, 
                    double* itersTriadTime,
                    double* totalTriadTime)
{
    int n = nItems;
    const float *restrict a[2];
    const float *restrict b[2];
    float *restrict c[2];
    a[0] = A; 
    a[1] = &A[n/2];
    b[0] = B; 
    b[1] = &B[n/2];
    c[0] = C; 
    c[1] = &C[n/2];

    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    // now that we've copied data to device,
    // time the iterations.
    int iterTimerHandle = Timer_Start();

    #pragma acc data create(a[0:2][0:(n/2)+n%2],b[0:2][0:(n/2)+n%2],c[0:2][0:(n/2)+n%2])
    // note: do *not* try to map the iterations loop to the accelerator
    for( unsigned int iter = 0; iter < nIters; iter++ )
    {
        //first stream
        #pragma acc update device(a[0][0:n/2],b[0][0:n/2]) async(1)
        #pragma acc kernels async(1)
        for( unsigned int i = 0; i < n/2; i++ )
        {
            c[0][i] = a[0][i] + s*b[0][i];
        }
        #pragma acc update host(c[0][0:n/2]) async(1)
        # pragma acc wait(1)

        //secondstream
        //+n%2 to pick up last element of odd n array
        #pragma acc update device(a[1][0:(n/2)+n%2],b[1][0:(n/2)+n%2]) async(2)
        #pragma acc kernels async(2)
        for( unsigned int i = 0; i < n/2+n%2; i++ )
        {
            c[1][i] = a[1][i] + s*b[1][i];
        }
        #pragma acc update host(c[1][0:(n/2)+n%2]) async(2)
        #pragma acc wait(2)
    }

    // stop the timer and record the result (in seconds)
    *itersTriadTime = Timer_Stop( iterTimerHandle, "" );

    *totalTriadTime = Timer_Stop( wholeTimerHandle, "" );


}

void 
DoTriadDoublesIters( unsigned int nIters, 
                     unsigned int nItems, 
                     double* A, 
                     double* B, 
                     double s, 
                     double* C, 
                     double* itersTriadTime,
                     double* totalTriadTime)
{

    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    // now that we've copied data to device,
    // time the iterations.
    int iterTimerHandle = Timer_Start();

    // note: do *not* try to map the iterations loop to the accelerator
    for( unsigned int iter = 0; iter < nIters; iter++ )
    {
        //# pragma acc data copyin(A[0:nItems],B[0:nItems]), copyout(C[0:nItems])
        #pragma acc data copyin(A[0:nItems],B[0:nItems]), copyout(C[0:nItems])
        {
            //# pragma acc kernels 
            #pragma acc kernels loop
            for( unsigned int i = 0; i < nItems; i++ )
            {
                C[i] = A[i] + s*B[i];
            }
        }
    }

    // stop the timer and record the result (in seconds)
    *itersTriadTime = Timer_Stop( iterTimerHandle, "" );

    *totalTriadTime = Timer_Stop( wholeTimerHandle, "" );

}

