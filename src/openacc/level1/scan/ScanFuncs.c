#include <stdio.h>
#include <stdlib.h>
#include "CTimer.h"




// Scan operation, using a naive (but GPU-accelerated) algorithm.
// Intended to show approachable use of OpenACC directives with 
// implementation of the naive sequential CPU-based algorithm.
void
DoNaiveScanFloatsIters( unsigned int nIters,
                        void* restrict ivdata, 
                        unsigned int nItems, 
                        void* restrict ovres, 
                        double* itersScanTime,
                        double* totalScanTime )
{
    float* restrict idata = (float*)ivdata;
    float* restrict ores = (float*)ovres;

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
DoNaiveScanDoublesIters( unsigned int nIters,
                        void* restrict ivdata, 
                        unsigned int nItems, 
                        unsigned int nBlocks,
                        void* restrict ovres, 
                        double* itersScanTime,
                        double* totalScanTime )
{
    double* restrict idata = (double*)ivdata;
    double* restrict ores = (double*)ovres;

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



void
DoScanFloatsIters( unsigned int nIters,
                        void* restrict ivdata, 
                        unsigned int nItems, 
                        unsigned int nBlocks,
                        void* restrict ovres, 
                        double* itersScanTime,
                        double* totalScanTime )
{
    float* restrict idata = (float*)ivdata;
    float* restrict odata = (float*)ovres;
    unsigned int iter;
    unsigned int itemIdx;
    unsigned int pairIdx;
    unsigned int nPairs;


    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    #pragma acc data copyin(idata[0:nItems]), copyout(odata[0:nItems])
    {
        // now that we've copied data to device,
        // time the iterations.
        int iterTimerHandle = Timer_Start();

        // note: do *not* try to map the iterations loop to the accelerator
        for( iter = 0; iter < nIters; iter++ )
        {
            unsigned int stride;

            // Copy the input to the output array (we do the prefix scan
            // in-place the output array, because we need to leave the 
            // input array unchanged for later verification).
            //
            // Note: if we were sure we were not going to need to the input
            // data after the function, we could do the parallel prefix
            // operation in place on the input array.
            #pragma acc kernels loop, present(idata[0:nItems], odata[0:nItems])
            for( itemIdx = 0; itemIdx < nItems; itemIdx++ )
            {
                odata[itemIdx] = idata[itemIdx];
            }

            //
            // Perform a Kogge-Stone prefix sum on the data
            // now held in the output array.
            //

            // Reduce phase
            for( stride = 1; stride < nItems; stride *= 2 )
            {
                #pragma acc kernels loop, independent, present(odata[0:nItems]), copyin(stride)
                for( pairIdx = 0; pairIdx < (nItems / (stride * 2)); pairIdx++ )
                {
                    unsigned int currIdx = ((pairIdx+1)*(stride*2)) - 1;
                    unsigned int combIdx = currIdx - stride;
                    odata[currIdx] += odata[combIdx];
                }
            }
            // fprintf( stderr, "after reduce, stride = %d\n", stride );

            // downsweep phase
            //
            // We start the stride a factor of four smaller than we
            // ended with the reduce phase loop.  The factor of four
            // comes from: a factor of two for the doubling that kicked us
            // out of the reduce loop, and a factor of two since we add
            // each element to the element halfway up to the next element
            // being added.
            for( stride = stride / 4; stride > 0; stride /= 2 )
            {
                unsigned int nPairs = (nItems / (stride * 4)) * 2 - 1;

                #pragma acc kernels loop, independent, present(odata[0:nItems]), copyin(stride)
                for( pairIdx = 0; pairIdx < nPairs; pairIdx++ )
                {
                    unsigned int combIdx = (pairIdx + 1) * (stride * 2) - 1;
                    unsigned int currIdx = combIdx + stride;
                    odata[currIdx] += odata[combIdx];
                }
            }
        }

        // stop the timer and record the result (in seconds)
        *itersScanTime = Timer_Stop( iterTimerHandle, "" );

    } /* end acc data region */


    *totalScanTime = Timer_Stop( wholeTimerHandle, "" );
}


void
DoScanDoublesIters( unsigned int nIters,
                        void* restrict ivdata, 
                        unsigned int nItems, 
                        unsigned int nBlocks,
                        void* restrict ovres, 
                        double* itersScanTime,
                        double* totalScanTime )
{
    double* restrict idata = (double*)ivdata;
    double* restrict odata = (double*)ovres;
    unsigned int iter;
    unsigned int itemIdx;
    unsigned int pairIdx;
    unsigned int nPairs;


    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    #pragma acc data copyin(idata[0:nItems]), copyout(odata[0:nItems])
    {
        // now that we've copied data to device,
        // time the iterations.
        int iterTimerHandle = Timer_Start();

        // note: do *not* try to map the iterations loop to the accelerator
        for( iter = 0; iter < nIters; iter++ )
        {
            unsigned int stride;

            // Copy the input to the output array (we do the prefix scan
            // in-place the output array, because we need to leave the 
            // input array unchanged for later verification).
            //
            // Note: if we were sure we were not going to need to the input
            // data after the function, we could do the parallel prefix
            // operation in place on the input array.
            #pragma acc kernels loop, present(idata[0:nItems], odata[0:nItems])
            for( itemIdx = 0; itemIdx < nItems; itemIdx++ )
            {
                odata[itemIdx] = idata[itemIdx];
            }

            //
            // Perform a Kogge-Stone prefix sum on the data
            // now held in the output array.
            //

            // Reduce phase
            for( stride = 1; stride < nItems; stride *= 2 )
            {
                #pragma acc kernels loop, independent, present(odata[0:nItems]), copyin(stride)
                for( pairIdx = 0; pairIdx < (nItems / (stride * 2)); pairIdx++ )
                {
                    unsigned int currIdx = ((pairIdx+1)*(stride*2)) - 1;
                    unsigned int combIdx = currIdx - stride;
                    odata[currIdx] += odata[combIdx];
                }
            }
            // fprintf( stderr, "after reduce, stride = %d\n", stride );

            // downsweep phase
            //
            // We start the stride a factor of four smaller than we
            // ended with the reduce phase loop.  The factor of four
            // comes from: a factor of two for the doubling that kicked us
            // out of the reduce loop, and a factor of two since we add
            // each element to the element halfway up to the next element
            // being added.
            for( stride = stride / 4; stride > 0; stride /= 2 )
            {
                unsigned int nPairs = (nItems / (stride * 4)) * 2 - 1;

                #pragma acc kernels loop, independent, present(odata[0:nItems]), copyin(stride)
                for( pairIdx = 0; pairIdx < nPairs; pairIdx++ )
                {
                    unsigned int combIdx = (pairIdx + 1) * (stride * 2) - 1;
                    unsigned int currIdx = combIdx + stride;
                    odata[currIdx] += odata[combIdx];
                }
            }
        }

        // stop the timer and record the result (in seconds)
        *itersScanTime = Timer_Stop( iterTimerHandle, "" );

    } /* end acc data region */


    *totalScanTime = Timer_Stop( wholeTimerHandle, "" );
}


