#include <stdio.h>
#include <stdlib.h>


/*
 * We use a 1D array to hold our matrix, which is logically 2D.
 * We use a helper to simplify the code that does indexing into
 * this 1D array.
 * We would prefer this to be a inline function, but PGI compiler
 * limitations restrict what can be done within an OpenACC kernels region,
 * and it does not like use of the inline function.  So we use a macro
 * instead.
 */
#define dval(a, nCols, i, j)    a[(i)*(nCols) + (j)]


/*
 * We would like to do this in C++, with a template function,
 * so that we don't have to reproduce the code differing only
 * in the matrix element data type (float vs double).  However,
 * since PGI's compiler does not support OpenACC directives in C++ code,
 * we implement these operations in C and do without the templates by
 * duplicating the code.
 */
void
ApplyFloatStencil( void* vdata,
                    unsigned int nRows,
                    unsigned int nCols,
                    unsigned int nPaddedCols,
                    unsigned int nIters,
                    unsigned int nItersPerExchange,
                    void* vwCenter,
                    void* vwCardinal,
                    void* vwDiagonal,
                    void (*preIterBlockCB)(void* cbData),
                    void* cbData )
{
    float* restrict data = (float*)vdata;
    float wCenter = *(float*)vwCenter;
    float wCardinal = *(float*)vwCardinal;
    float wDiagonal = *(float*)vwDiagonal;


    /*
     * Our algorithm is float buffering.  We need to allocate a buffer
     * on the device of the same size as the input data.  We use
     * OpenACC's create clause on a data region to accomplish this.
     */
    float* restrict other = (float*)calloc( nRows * nPaddedCols, sizeof(float) );

    #pragma acc data create(other[0:nRows*nPaddedCols])
    {

    /* Perform the stencil operation for the desired number of iterations.
     * To support the necessary halo exchanges in the truly parallel version,
     * we need to ensure that the data in the "data" matrix is valid in
     * the host memory every "nItersPerExchange" iterations.  Since OpenACC
     * doesn't give us an explicit operation to "read data from device now"
     * as we have with OpenCL and CUDA, we have to break up the iterations
     * and put each block of iterations between exchanges into its own
     * data region for the "data" array.
     *
     * For the sequential version, this logic should degenerate to a
     * single data region for "data," as long as nIters == nItersPerExchange.
     */
    unsigned int nIterBlocks = (nIters / nItersPerExchange) +
        ((nIters % nItersPerExchange) ? 1 : 0);

    for( unsigned int iterBlockIdx = 0; iterBlockIdx < nIterBlocks; iterBlockIdx++ )
    {
        unsigned int iterLow = iterBlockIdx * nItersPerExchange;
        unsigned int iterHighBound = (iterBlockIdx + 1) * nItersPerExchange;
        if( iterHighBound > nIters )
        {
            iterHighBound = nIters;
        }

        /* do any per-iteration-block work (e.g., do a halo exchange!) */
        if( preIterBlockCB != NULL )
        {
            (*preIterBlockCB)( cbData );
        }

        #pragma acc data present_or_copy(data[0:nRows*nPaddedCols])
        {

        for( unsigned int iter = iterLow; iter < iterHighBound; iter++ )
        {
            /* apply the stencil operator */
            #pragma acc kernels loop independent
            for( unsigned int i = 1; i < (nRows-1); i++ )
            {
                #pragma acc loop independent
                for( unsigned int j = 1; j < (nPaddedCols-1); j++ )
                {
                    float oldCenterValue = dval(data, nPaddedCols, i, j);
                    float oldNSEWValues = dval(data, nPaddedCols, i - 1, j ) +
                                            dval(data, nPaddedCols, i + 1, j ) +
                                            dval(data, nPaddedCols, i, j - 1 ) +
                                            dval(data, nPaddedCols, i, j + 1 );
                    float oldDiagonalValues = dval(data, nPaddedCols, i - 1, j - 1) +
                                                dval(data, nPaddedCols, i - 1, j + 1) +
                                                dval(data, nPaddedCols, i + 1, j - 1) +
                                                dval(data, nPaddedCols, i + 1, j + 1);

                    float newVal = wCenter * oldCenterValue +
                                    wCardinal * oldNSEWValues +
                                    wDiagonal * oldDiagonalValues;
                    dval(other, nPaddedCols, i, j ) = newVal;
                }
            }

            /* Copy the new values into the "real" array (data)
             * Note: we would like to just swap pointers between a "current" 
             * and "new" array, but have not figured out how to do this successfully
             * within OpenACC kernels region.
             */
            #pragma acc kernels loop independent
            for( unsigned int i = 1; i < (nRows - 1); i++ )
            {
                #pragma acc loop independent
                for( unsigned int j = 1; j < (nCols - 1); j++ )
                {
                    dval(data, nPaddedCols, i, j) = dval(other, nPaddedCols, i, j);
                }
            }
        }

        } /* end of OpenACC data region for "data" array */
    }

    } /* end of OpenACC data region for "other" array */

    free(other);
}

void
ApplyDoubleStencil( void* vdata,
                    unsigned int nRows,
                    unsigned int nCols,
                    unsigned int nPaddedCols,
                    unsigned int nIters,
                    unsigned int nItersPerExchange,
                    void* vwCenter,
                    void* vwCardinal,
                    void* vwDiagonal,
                    void (*preIterBlockCB)(void* cbData),
                    void* cbData )
{
    double* restrict data = (double*)vdata;
    double wCenter = *(double*)vwCenter;
    double wCardinal = *(double*)vwCardinal;
    double wDiagonal = *(double*)vwDiagonal;


    /*
     * Our algorithm is double buffering.  We need to allocate a buffer
     * on the device of the same size as the input data.  We use
     * OpenACC's create clause on a data region to accomplish this.
     */
    double* restrict other = (double*)calloc( nRows * nPaddedCols, sizeof(double) );

    #pragma acc data create(other[0:nRows*nPaddedCols])
    {

    /* Perform the stencil operation for the desired number of iterations.
     * To support the necessary halo exchanges in the truly parallel version,
     * we need to ensure that the data in the "data" matrix is valid in
     * the host memory every "nItersPerExchange" iterations.  Since OpenACC
     * doesn't give us an explicit operation to "read data from device now"
     * as we have with OpenCL and CUDA, we have to break up the iterations
     * and put each block of iterations between exchanges into its own
     * data region for the "data" array.
     *
     * For the sequential version, this logic should degenerate to a
     * single data region for "data," as long as nIters == nItersPerExchange.
     */
    unsigned int nIterBlocks = (nIters / nItersPerExchange) +
        ((nIters % nItersPerExchange) ? 1 : 0);

    for( unsigned int iterBlockIdx = 0; iterBlockIdx < nIterBlocks; iterBlockIdx++ )
    {
        unsigned int iterLow = iterBlockIdx * nItersPerExchange;
        unsigned int iterHighBound = (iterBlockIdx + 1) * nItersPerExchange;
        if( iterHighBound > nIters )
        {
            iterHighBound = nIters;
        }

        /* do any per-iteration-block work (e.g., do a halo exchange!) */
        if( preIterBlockCB != NULL )
        {
            (*preIterBlockCB)( cbData );
        }

        #pragma acc data present_or_copy(data[0:nRows*nPaddedCols])
        {

        for( unsigned int iter = iterLow; iter < iterHighBound; iter++ )
        {
            /* apply the stencil operator */
            #pragma acc kernels loop independent
            for( unsigned int i = 1; i < (nRows-1); i++ )
            {
                #pragma acc loop independent
                for( unsigned int j = 1; j < (nPaddedCols-1); j++ )
                {
                    double oldCenterValue = dval(data, nPaddedCols, i, j);
                    double oldNSEWValues = dval(data, nPaddedCols, i - 1, j ) +
                                            dval(data, nPaddedCols, i + 1, j ) +
                                            dval(data, nPaddedCols, i, j - 1 ) +
                                            dval(data, nPaddedCols, i, j + 1 );
                    double oldDiagonalValues = dval(data, nPaddedCols, i - 1, j - 1) +
                                                dval(data, nPaddedCols, i - 1, j + 1) +
                                                dval(data, nPaddedCols, i + 1, j - 1) +
                                                dval(data, nPaddedCols, i + 1, j + 1);

                    double newVal = wCenter * oldCenterValue +
                                    wCardinal * oldNSEWValues +
                                    wDiagonal * oldDiagonalValues;
                    dval(other, nPaddedCols, i, j ) = newVal;
                }
            }

            /* Copy the new values into the "real" array (data)
             * Note: we would like to just swap pointers between a "current" 
             * and "new" array, but have not figured out how to do this successfully
             * within OpenACC kernels region.
             */
            #pragma acc kernels loop independent
            for( unsigned int i = 1; i < (nRows - 1); i++ )
            {
                #pragma acc loop independent
                for( unsigned int j = 1; j < (nCols - 1); j++ )
                {
                    dval(data, nPaddedCols, i, j) = dval(other, nPaddedCols, i, j);
                }
            }
        }

        } /* end of OpenACC data region for "data" array */
    }

    } /* end of OpenACC data region for "other" array */

    free(other);
}

