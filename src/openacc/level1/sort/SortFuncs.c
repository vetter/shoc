#include <stdio.h>
#include <stdlib.h>

#include "CTimer.h"

#define BASE 16             //number of "buckets" to use
#define NTHREADS (2*1024)   //number of threads to use

void radixSort(uint *restrict hKeys, int size, double *SortTime, double *TransferTime)
{
    unsigned int restrict* tmpKeys; 
    int i,j, pass, thread, npt;
    unsigned int mask = BASE-1, temp, temp1;

    int n = size * sizeof(uint); //number of elements to sort
    npt = n / NTHREADS;  // number per thread

    tmpKeys = (unsigned int *)malloc(n*sizeof(unsigned int));

    // start a timer 
    // need OpenACC profiling support for accurate transfer times
    // until then, just set and leave it at zero
    int SortTimerHandle;
    int TransferTimerHandle;

    *SortTime = 0.0;
    *TransferTime = 0.0;
    for(pass = 0; pass < 8; pass++) //let's deal with 8bits at a time
    {
       mask = 0;
       mask = BASE-1 << 4*pass;
    
        // "shared" variables
        unsigned int sums[NTHREADS] = {0};
        unsigned int sumsb[NTHREADS] = {0};
        unsigned int count[NTHREADS*BASE] = {0};
    
        #pragma acc data copy(hKeys[0:n]), copyin(count[0:NTHREADS*BASE]), create(tmpKeys[0:n])
        {

            SortTimerHandle = Timer_Start();
            /* divide the input array into nthreads chunks */
            #pragma acc kernels loop independent
            for(thread = 0; thread < NTHREADS; thread++)
            {
                for (i = 0; i < npt; i++) //Count the number of keys for each bin
                {
                    temp = (hKeys[thread*npt + i] & mask) >> 4*pass;
                    count[temp*NTHREADS + thread]++;  //"column-major" order 
                                                      //for easier summing later 
                }
            }
            *SortTime += Timer_Stop( SortTimerHandle, "" );
        
        #pragma acc data copy(sums,sumsb)
        {

            SortTimerHandle = Timer_Start();
            //Parallel scan to get the offset locations in data memory
            #pragma acc parallel loop
            for(thread = 0; thread < NTHREADS; thread++)
            {
                sums[thread] = 0;
                sumsb[thread] = 0;
                for(i=0; i < BASE; i++) 
                {
                    sums[thread] += count[BASE*thread + i]; //sum first
                }
            }
            #pragma acc kernels loop independent
            for(thread = 0; thread < NTHREADS; thread++)
            {
                temp1 = count[BASE*thread + 0];
                count[BASE*thread + 0] = 0;
                for (i = 1; i < BASE; i++) 
                {
                    temp = count[BASE*thread + i];
                    count[BASE*thread + i  ] = count[BASE*thread + i-1] + temp1;
                    temp1 = temp;
                }
            } 
        
        
            //Parallel prefix sum the sums
            //first, shift to the right
            #pragma acc kernels loop
            for(thread = NTHREADS-1; thread > 0; thread--)  
            {
                sumsb[thread] = sums[thread-1];  
            }
            #pragma acc kernels loop
            for(thread = 0; thread < NTHREADS; thread++) 
            {
                sums[thread] = sumsb[thread];
            }
            *SortTime += Timer_Stop( SortTimerHandle, "" );
        
        } //end sums,sumsb acc data copy
        
            sums[0] = 0; 
        
            SortTimerHandle = Timer_Start();
            //parallel reduction (sum) 
            for (i = 1; i <= NTHREADS/2; i = i*2) 
            {
                for(thread = i; thread < NTHREADS; thread++) 
                { 
                    sumsb[thread] = sums[thread-i] + sums[thread]; 
                }
                for(thread = i; thread < NTHREADS; thread++) 
                { 
                    sums[thread] = sumsb[thread]; 
                }
            }
        
            #pragma acc kernels loop copyin(sums)
            //Add each thread's prefix sums to sums to get global offsets
            for (i = 0; i < BASE; i++) //FIXME reversing loop order speeds us up like crazy?
            {
                for(thread = 0; thread < NTHREADS; thread++) 
                {
                    count[BASE*thread + i] += sums[thread]; 
                }
            }
        
            //Place keys into offsets 
            #pragma acc kernels loop independent
            for(thread = 0; thread < NTHREADS; thread++) 
            {
                for (i = 0; i < npt; i++)
                {
                    temp = (hKeys[thread*npt + i] & mask) >> 4*pass;
                    tmpKeys[count[temp*NTHREADS + thread]] = hKeys[thread*npt + i];
                    count[temp*NTHREADS + thread]++; //in case multiple keys for this slot
                }
            }
        
            //Copy array tmpKeys to array hKeys
            #pragma acc kernels loop independent
            for (i = 0; i < npt; i++) //FIXME again with the loop ordering speedup?
            {
                for(thread = 0; thread < NTHREADS; thread++) 
                {
                    hKeys[thread*npt + i] = tmpKeys[thread*npt + i];
                }
            }
            *SortTime += Timer_Stop( SortTimerHandle, "" );
        
        } //END acc data region
    }

}
