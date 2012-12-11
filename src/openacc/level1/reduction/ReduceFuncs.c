#include <stdio.h>
#include "CTimer.h"



void
DoReduceFloatsIters( unsigned int nIters,
                        void* ivdata, 
                        unsigned int nItems, 
                        void* ovres, 
                        double* totalReduceTime,
                        void (*reducefunc)( void* localsum, void* result ) )
{
    float sum = 0.0;
    float* restrict idata = (float*)ivdata;
    float* ores = (float*)ovres;

    #pragma acc data pcopyin(idata[0:nItems])
    {

        // now that we've copied data to device,
        // time the iterations.
        int iterTimerHandle = Timer_Start();

        // note: do *not* try to map the iterations loop to the accelerator
        for( unsigned int iter = 0; iter < nIters; iter++ )
        {
            sum = 0.0;

            #pragma acc kernels loop reduction( +:sum ) independent 
            for( unsigned int i = 0; i < nItems; i++ )
            {
                sum += idata[i];
            }

            // we may have to reduce further
            if( reducefunc != 0 )
            {
                float res;
                (*reducefunc)( &sum, &res );
                sum = res;
            }
        }

        // stop the timer and record the result (in seconds)
        *totalReduceTime = Timer_Stop( iterTimerHandle, "" );

    } /* end acc data region for idata */

    // save the result
    *ores = sum;
}




void
DoReduceDoublesIters( unsigned int nIters,
                        void* ivdata, 
                        unsigned int nItems, 
                        void* ovres, 
                        double* totalReduceTime,
                        void (*reducefunc)( void* localsum, void* result ) )
{
    double sum = 0.0;
    double* restrict idata = (double*)ivdata;
    double* ores = (double*)ovres;

    #pragma acc data pcopyin(idata[0:nItems])
    {

        // now that we've copied data to device,
        // time the iterations.
        int iterTimerHandle = Timer_Start();

        // note: do *not* try to map the iterations loop to the accelerator
        for( unsigned int iter = 0; iter < nIters; iter++ )
        {
            sum = 0.0;

            #pragma acc kernels loop reduction( +:sum ) independent
            for( unsigned int i = 0; i < nItems; i++ )
            {
                sum += idata[i];
            }

            // we may have to reduce further
            if( reducefunc != 0 )
            {
                double res;
                (*reducefunc)( &sum, &res );
                sum = res;
            }
        }

        // stop the timer and record the result (in seconds)
        *totalReduceTime = Timer_Stop( iterTimerHandle, "" );

    } /* end acc data region for idata */

    // save the result
    *ores = sum;
}



