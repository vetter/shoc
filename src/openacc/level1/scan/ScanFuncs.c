#include <stdio.h>
#include <stdlib.h>
#include "CTimer.h"



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
    float* restrict block_sums = (float*)malloc( nBlocks * sizeof(float) );
    float* restrict top_scans = (float*)malloc( nBlocks * sizeof(float) );
    unsigned int iter;
    unsigned int blockSize = nItems / nBlocks;
    unsigned int blockIndex;
    unsigned int itemIdx;


    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    #pragma acc data pcopyin(idata[0:nItems]), pcopyout(odata[0:nItems]), create(block_sums[0:nBlocks], top_scans[0:nBlocks])
    {
        // now that we've copied data to device,
        // time the iterations.
        int iterTimerHandle = Timer_Start();

        // note: do *not* try to map the iterations loop to the accelerator
        for( iter = 0; iter < nIters; iter++ )
        {
            // reduce - find the sum of values within each block
            #pragma acc parallel loop num_gangs(nBlocks)
            for( blockIndex = 0; blockIndex < nBlocks; blockIndex++ )
            {
                float sum = 0.0f;

                #pragma acc loop reduction(+:sum)
                for( itemIdx = 0; itemIdx < blockSize; itemIdx++ )
                {
                    sum += idata[(blockIndex * blockSize) + itemIdx];
                }
                block_sums[blockIndex] = sum;
            }

            // top_scan - do a parallel prefix (scan) operation over the
            // block sums
            // This scan is offset by 1 element, because we need the
            // bottom_scan of the first block to start with zero, 
            // the second block to start with the sum of values from 
            // the first block, etc.
            float runningVal = 0.0f;

            #pragma acc kernels
            for( blockIndex = 0; blockIndex < nBlocks; blockIndex++ )
            {
                top_scans[blockIndex] = runningVal;
                runningVal = runningVal + block_sums[blockIndex];
            }

            // bottom_scan - for each block, start with the block's 
            // block sum value and the find the parallel prefix for values
            // within the block
            #pragma acc parallel loop num_gangs(nBlocks), pcopyin(top_scans[0:nBlocks])
            for( blockIndex = 0; blockIndex < nBlocks; blockIndex++ )
            {
                float brunningVal = top_scans[blockIndex];

                // PGI 12.10 thinks this loop is parallelizable.
                // But it has a loop carried dependence on brunningVal
                // (it is both read and written within each iteration of
                // the loop) so the 
                #pragma acc loop seq
                for( itemIdx = 0; itemIdx < blockSize; itemIdx++ )
                {
                    unsigned int currIdx = (blockIndex*blockSize) + itemIdx;

                    odata[currIdx] = idata[currIdx] + brunningVal;
                    brunningVal = odata[currIdx];
                }
            }
        }

        // stop the timer and record the result (in seconds)
        *itersScanTime = Timer_Stop( iterTimerHandle, "" );

    } /* end acc data region for idata */

    // Handle any leftovers if nBlocks does not divide nItems evenly
    // Do this on the CPU to avoid complex array bounds checking logic
    // within OpenACC regions that the PGI 12.10 compiler did not seem 
    // to handle well.
    for( itemIdx = nBlocks*blockSize; itemIdx < nItems; itemIdx++ )
    {
        odata[itemIdx] = idata[itemIdx] + odata[itemIdx-1];
    }

    *totalScanTime = Timer_Stop( wholeTimerHandle, "" );

    free( block_sums );
    free( top_scans );
}




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
DoScanDoublesIters( unsigned int nIters,
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



