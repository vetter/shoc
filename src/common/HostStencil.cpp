#include <string.h> // for memcpy
#include "HostStencil.h"


template<class T>
void
HostStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    // we need a temp space buffer
    Matrix2D<T> tmpMtx( mtx.GetNumRows(), mtx.GetNumColumns() );

    // be able to access the matrices as 2D arrays
    typename Matrix2D<T>::DataPtr mtxData = mtx.GetData();
    typename Matrix2D<T>::DataPtr tmpMtxData = tmpMtx.GetData();


    for( unsigned int iter = 0; iter < nIters; iter++ )
    {
        DoPreIterationWork( mtx, iter );

        /* copy the "real" data to the temp matrix */
        memcpy( tmpMtx.GetFlatData(),
                mtx.GetFlatData(),
                mtx.GetDataSize() );


        /* Apply the stencil operator */
        for( size_t i = 1; i < mtx.GetNumRows()-1; i++ )
        {
            for( size_t j = 1; j < mtx.GetNumColumns()-1; j++ )
            {
                T oldCenterValue = tmpMtxData[i][j];
                T oldNSEWValues = (tmpMtxData[i-1][j] +
                                        tmpMtxData[i+1][j] +
                                        tmpMtxData[i][j-1] +
                                        tmpMtxData[i][j+1]);
                T oldDiagonalValues = (tmpMtxData[i-1][j-1] +
                                            tmpMtxData[i+1][j-1] +
                                            tmpMtxData[i-1][j+1] +
                                            tmpMtxData[i+1][j+1]);

                mtxData[i][j] = this->wCenter * oldCenterValue +
                                this->wCardinal * oldNSEWValues +
                                this->wDiagonal * oldDiagonalValues;
            }
        }
    }
}


template<class T>
void
HostStencil<T>::DoPreIterationWork( Matrix2D<T>& mtx, unsigned int iter )
{
    // we have nothing to do
}

