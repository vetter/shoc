#include <stdio.h>
#include "CTimer.h"



void
DoScanFloatsIters( unsigned int nIters,
                        void* ivdata, 
                        unsigned int nItems, 
                        void* ovres, 
                        double* itersScanTime,
                        double* totalScanTime )
{
    float* restrict idata = (float*)ivdata;
    float* ores = (float*)ovres;

    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    #pragma acc data pcopyin(idata[0:nItems]), pcopyout(ores[0:nItems])
    {

        // now that we've copied data to device,
        // time the iterations.
        int iterTimerHandle = Timer_Start();

        // note: do *not* try to map the iterations loop to the accelerator
        for( unsigned int iter = 0; iter < nIters; iter++ )
        {
            float runningVal = 0.0f;

            #pragma acc kernels loop
            for( unsigned int i = 0; i < nItems; i++ )
            {
                ores[i] = idata[i] + runningVal;
                runningVal = ores[i];
            }
        }

        // stop the timer and record the result (in seconds)
        *itersScanTime = Timer_Stop( iterTimerHandle, "" );

    } /* end acc data region for idata */

    *totalScanTime = Timer_Stop( wholeTimerHandle, "" );
}




void
DoScanDoublesIters( unsigned int nIters,
                        void* ivdata, 
                        unsigned int nItems, 
                        void* ovres, 
                        double* itersScanTime,
                        double* totalScanTime )
{
    double* restrict idata = (double*)ivdata;
    double* ores = (double*)ovres;

    // start a timer that includes both transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    #pragma acc data pcopyin(idata[0:nItems]), pcopyout(ores[0:nItems])
    {
        // now that we've copied data to device,
        // time the iterations.
        int iterTimerHandle = Timer_Start();

        // note: do *not* try to map the iterations loop to the accelerator
        for( unsigned int iter = 0; iter < nIters; iter++ )
        {
            double runningVal = 0.0;

            #pragma acc kernels loop
            for( unsigned int i = 0; i < nItems; i++ )
            {
                ores[i] = idata[i] + runningVal;
                runningVal = ores[i];
            }
        }

        // stop the timer and record the result (in seconds)
        *itersScanTime = Timer_Stop( iterTimerHandle, "" );

    } /* end acc data region for idata */

    *totalScanTime = Timer_Stop( wholeTimerHandle, "" );
}



