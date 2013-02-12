#include <stdio.h>
#include <stdlib.h>
#include "CTimer.h"

void DummyFloatFunc( float v );
void DummyDoubleFunc( float v );


void
DoScanFloatsIters( unsigned int nIters,
                        void* restrict ivdata, 
                        unsigned int nItems, 
                        void* restrict ovres, 
                        double* itersScanTime,
                        double* totalScanTime,
                        void (*gscanFunc)(void*, void*) )
{
    float* restrict idata = (float*)ivdata;
    float* restrict odata = (float*)ovres;
    unsigned int iter;
    unsigned int itemIdx;
    unsigned int pairIdx;
    unsigned int nPairs;
    float gBaseValue = 0.0f;


    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    #pragma acc data pcopyin(idata[0:nItems]), copyout(odata[0:nItems])
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
            // #pragma acc kernels loop, present(idata[0:nItems], odata[0:nItems])
            #pragma acc kernels loop
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
                #pragma acc kernels loop, independent
                for( pairIdx = 0; pairIdx < (nItems / (stride * 2)); pairIdx++ )
                {
                    unsigned int currIdx = ((pairIdx+1)*(stride*2)) - 1;
                    unsigned int combIdx = currIdx - stride;
                    odata[currIdx] += odata[combIdx];
                }
            }
            // fprintf( stderr, "after reduce, stride = %d\n", stride );

            // If we are participating in a Truly Parallel scan 
            // (across multiple MPI tasks), we need to do a scan 
            // across all tasks using the sum of the values of each
            // task as their contribution to the scan.  Later, we will
            // add the value we get from this scan to our local scan
            // values to get our values for the global scan.
            // Happily, we have the sum of our local values from the 
            // Reduce phase, in the last element of the odata array.
            if( gscanFunc != 0 )
            {
                #pragma acc update host(odata[nItems-1:1])
                (*gscanFunc)( &(odata[nItems-1]), &gBaseValue );

                // The PGI compiler, as of version 12.10, 
                // does not seem to respect our copyout clause.
                // Perhaps it thinks lReduceValue is dead in the host code, 
                // and does not bother to copy it off the device.
                //
                // Adding a printf here seems to get the compiler to 
                // copy the value off the device, but we would really
                // prefer not to dump output here.  Instead, we use
                // a dummy function implemented in another file, so that
                // the compiler doesn't know that it does not use the
                // variable's value.  This approach will break if the
                // compiler is smart enough on its interprocedural
                // optimization.
                //
                // Note all of it is unnecessary if the compiler would
                // respect our copyout clause.
                // DummyFloatFunc( lReduceValue );

                /*
                fprintf( stderr, "lReduceValue=%lf, gBaseValue=%lf\n", 
                    lReduceValue,
                    gBaseValue );
                */
            }


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

                #pragma acc kernels loop, independent
                for( pairIdx = 0; pairIdx < nPairs; pairIdx++ )
                {
                    unsigned int combIdx = (pairIdx + 1) * (stride * 2) - 1;
                    unsigned int currIdx = combIdx + stride;
                    odata[currIdx] += odata[combIdx];
                }
            }

            if( gscanFunc != 0 )
            {
                // We are performing a TP Scan operation
                // We need to add the base value we got from our global scan
                // to each of the values produced by our local scan.
                #pragma acc kernels loop, independent, present(odata[0:nItems]), copyin(gBaseValue)
                for( itemIdx = 0; itemIdx < nItems; itemIdx++ )
                {
                    odata[itemIdx] += gBaseValue;                    
                }
            }
        }

        // stop the timer and record the result (in seconds)
        *itersScanTime = Timer_Stop( iterTimerHandle, "" );

    } /* end of data region */

    *totalScanTime = Timer_Stop( wholeTimerHandle, "" );
}


void
DoScanDoublesIters( unsigned int nIters,
                        void* restrict ivdata, 
                        unsigned int nItems, 
                        void* restrict ovres, 
                        double* itersScanTime,
                        double* totalScanTime,
                        void (*gscanFunc)(void*, void*) )
{
    double* restrict idata = (double*)ivdata;
    double* restrict odata = (double*)ovres;
    unsigned int iter;
    unsigned int itemIdx;
    unsigned int pairIdx;
    unsigned int nPairs;
    double lReduceValue = 0.0;
    double gBaseValue = 0.0;


    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    #pragma acc data pcopyin(idata[0:nItems]), copyout(odata[0:nItems])
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

            // If we are participating in a Truly Parallel scan 
            // (across multiple MPI tasks), we need to do a scan 
            // across all tasks using the sum of the values of each
            // task as their contribution to the scan.  Later, we will
            // add the value we get from this scan to our local scan
            // values to get our values for the global scan.
            // Happily, we have the sum of our local values from the 
            // Reduce phase, in the last element of the odata array.
            if( gscanFunc != 0 )
            {
                #pragma acc kernels copyout(lReduceValue)
                {
                    lReduceValue = odata[nItems - 1];
                }
                (*gscanFunc)( &lReduceValue, &gBaseValue );

                // The PGI compiler, as of version 12.10, 
                // does not seem to respect our copyout clause.
                // Perhaps it thinks lReduceValue is dead in the host code, 
                // and does not bother to copy it off the device.
                //
                // Adding a printf here seems to get the compiler to 
                // copy the value off the device, but we would really
                // prefer not to dump output here.  Instead, we use
                // a dummy function implemented in another file, so that
                // the compiler doesn't know that it does not use the
                // variable's value.  This approach will break if the
                // compiler is smart enough on its interprocedural
                // optimization.
                //
                // Note all of it is unnecessary if the compiler would
                // respect our copyout clause.
                DummyDoubleFunc( lReduceValue );

                /*
                fprintf( stderr, "lReduceValue=%lf, gBaseValue=%lf\n", 
                    lReduceValue,
                    gBaseValue );
                */
            }


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

            if( gscanFunc != 0 )
            {
                // We are performing a TP Scan operation
                // We need to add the base value we got from our global scan
                // to each of the values produced by our local scan.
                #pragma acc kernels loop, independent, present(odata[0:nItems]), copyin(gBaseValue)
                for( itemIdx = 0; itemIdx < nItems; itemIdx++ )
                {
                    odata[itemIdx] += gBaseValue;                    
                }
            }
        }

        // stop the timer and record the result (in seconds)
        *itersScanTime = Timer_Stop( iterTimerHandle, "" );

    } /* end of data region */

    *totalScanTime = Timer_Stop( wholeTimerHandle, "" );
}


