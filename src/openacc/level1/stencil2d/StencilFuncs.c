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
ApplyDoubleStencil( void* vdata,
                    unsigned int nRows,
                    unsigned int nCols,
                    unsigned int nPaddedCols,
                    unsigned int nIters,
                    void* vwCenter,
                    void* vwCardinal,
                    void* vwDiagonal )
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
    double* restrict other = (double*)malloc( nRows * nPaddedCols * sizeof(double) );

#pragma acc data copy(data[0:nRows*nPaddedCols]) create(other[0:nRows*nPaddedCols])
    {

    /* perform the stencil operation for the desired number of iterations */
    for( unsigned int iter = 0; iter < nIters; iter++ )
    {
#if READY
        // do pre-iteration work - how to implement - via func pointer?
#endif // READY

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

    } /* end of OpenACC data region - matrix "data" is copied back to host */

    free(other);
}



#if READY
void
ApplyFloatStencil( void* vdata,
                    unsigned int nRows,
                    unsigned int nCols,
                    unsigned int nPaddedCols,
                    unsigned int nIters,
                    void* vwCenter,
                    void* vwCardinal,
                    void* vwDiagonal )
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
    float* restrict other = (float*)malloc( nRows * nPaddedCols * sizeof(float) );

#pragma acc data copy(data[0:nRows*nPaddedCols]) create (other[0:nRows*nPaddedCols])
    {

    /* perform the stencil operation for the desired number of iterations */
    for( unsigned int iter = 0; iter < nIters; iter++ )
    {
#if READY
        // do pre-iteration work - how to implement - via func pointer?
#endif // READY

        /* apply the stencil operator */
#pragma acc kernels loop independent
        for( unsigned int i = 1; i < (nRows-1); i++ )
        {
#pragma acc loop independent
            for( unsigned int j = 1; j < (nCols-1); j++ )
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
    }

    /* Copy the new values into the "real" array (data)
     * Note: we would like to just swap pointers between a "current" 
     * and "new" array, but have not figured out how to do this successfully
     * within OpenACC kernels region.
     */
#pragma acc kernels loop independent
    for( unsigned int i = 0; i < nRows; i++ )
    {
#pragma acc loop independent
        for( unsigned int j = 0; j < nCols; j++ )
        {
            dval(data, nPaddedCols, i, j) = dval(other, nPaddedCols, i, j);
        }
    }

    } /* end of OpenACC data region - matrix "data" is copied back to host */

    free(other);
}


#else

void
ApplyFloatStencil( void* vdata,
                    unsigned int nRows,
                    unsigned int nCols,
                    unsigned int nPaddedCols,
                    unsigned int nIters,
                    void* vwCenter,
                    void* vwCardinal,
                    void* vwDiagonal )
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
    float* restrict other = (float*)malloc( nRows * nPaddedCols * sizeof(float) );

#pragma acc data copy(data[0:nRows*nPaddedCols]) create(other[0:nRows*nPaddedCols])
    {

    /* perform the stencil operation for the desired number of iterations */
    for( unsigned int iter = 0; iter < nIters; iter++ )
    {
#if READY
        // do pre-iteration work - how to implement - via func pointer?
#endif // READY

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

    } /* end of OpenACC data region - matrix "data" is copied back to host */

    free(other);
}

#endif 
