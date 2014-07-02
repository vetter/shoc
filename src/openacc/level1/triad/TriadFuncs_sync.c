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
    //mgl don't we need restrict somewhere?

    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    // now that we've copied data to device,
    // time the iterations.
    int iterTimerHandle = Timer_Start();

    #pragma acc data create(A[0:nItems],B[0:nItems],C[0:nItems])
    // note: do *not* try to map the iterations loop to the accelerator
    for( unsigned int iter = 0; iter < nIters; iter++ )
    {
        //# pragma acc data async copyin(A[0:nItems],B[0:nItems]), copyout(C[0:nItems])
        //# pragma acc data copy(A[0:nItems],B[0:nItems]),copyout(C[0:nItems]) async(iter)
        {
            #pragma acc update device(A[0:nItems],B[0:nItems])
            
            #pragma acc kernels
            for( unsigned int i = 0; i < nItems; i++ )
            {
                C[i] = A[i] + s*B[i];
            }
            //# pragma acc wait(iter)
            #pragma acc update host(C[0:nItems])
        }
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

