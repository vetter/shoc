#include <stdio.h>
#include <stdlib.h>

#include "CTimer.h"

void 
DoTriadFloats( unsigned int nItems, 
               unsigned int blkSize, 
               const float *restrict A, 
               const float *restrict B, 
               float s, 
               float *restrict C, 
               double* totalTriadTime)
{

    int nChunks = nItems/blkSize;
    int i,j;

    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    // note: do *not* try to map the iterations loop to the accelerator
    #pragma acc data create(A[0:nItems],B[0:nItems],C[0:nItems])
    for( j = 0; j < nChunks; j++ )
    {
        #pragma acc update device(A[j*blkSize:blkSize],B[j*blkSize:blkSize]) async(j+1)
        #pragma acc parallel loop present(A[j*blkSize:blkSize],B[j*blkSize:blkSize],C[j*blkSize:blkSize]) async(j+1)
        for( i=0; i < blkSize; i++ )
        {
            C[j*blkSize + i] = A[j*blkSize + i] + s*B[j*blkSize + i];
        }
        #pragma acc update host(C[j*blkSize:blkSize]) async(j+1) //doubled this call to get full overlap of transfers
        #pragma acc update host(C[j*blkSize:blkSize]) async(j+1) 

    }
    #pragma acc wait 

    // stop the timer and record the result (in seconds)
    *totalTriadTime = Timer_Stop( wholeTimerHandle, "" );


}

void 
DoTriadDoubles ( unsigned int nItems, 
                 unsigned int blkSize, 
                 const double *restrict A, 
                 const double *restrict B, 
                 double s, 
                 double *restrict C, 
                 double* totalTriadTime)
{

    int nChunks = nItems/blkSize;
    int i,j;

    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    // note: do *not* try to map the iterations loop to the accelerator
    #pragma acc data create(A[0:nItems],B[0:nItems],C[0:nItems])
    for( j = 0; j < nChunks; j++ )
    {
        #pragma acc update device(A[j*blkSize:blkSize],B[j*blkSize:blkSize]) async(j+1)
        #pragma acc parallel loop present(A[j*blkSize:blkSize],B[j*blkSize:blkSize],C[j*blkSize:blkSize]) async(j+1)
        for( i=0; i < blkSize; i++ )
        {
            C[j*blkSize + i] = A[j*blkSize + i] + s*B[j*blkSize + i];
        }
        #pragma acc update host(C[j*blkSize:blkSize]) async(j+1) //doubled this call to get full overlap of transfers
        #pragma acc update host(C[j*blkSize:blkSize]) async(j+1) 

    }
    #pragma acc wait

    // stop the timer and record the result (in seconds)
    *totalTriadTime = Timer_Stop( wholeTimerHandle, "" );


}
