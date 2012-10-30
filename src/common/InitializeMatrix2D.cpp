#include <stdlib.h>
#include <string.h>
#include <cassert>
#include "InitializeMatrix2D.h"
#include "Utility.h"


template<class T>
void
Initialize<T>::operator()( Matrix2D<T>& mtx )
{
    srand48( seed );

    int nTileRows = mtx.GetNumRows() - 2 * haloWidth;
    if( (rowPeriod != -1) && (rowPeriod < nTileRows) )
    {
        nTileRows = rowPeriod;
    }

    int nTileCols = mtx.GetNumColumns() - 2 * haloWidth;
    if( (colPeriod != -1) && (colPeriod < nTileCols) )
    {
        nTileCols = colPeriod;
    }


    // initialize first tile
    for( unsigned int i = 0; i < nTileRows; i++ )
    {
        for( unsigned int j = 0; j < nTileCols; j++ )
        {
#ifndef READY
            mtx.GetData()[i+haloWidth][j+haloWidth] = i * j;
#else
            mtx.GetData()[i+haloWidth][j+haloWidth] = (T)drand48();
#endif // READY
        }
    }

    // initialize any remaining tiles
    // first we fill along rows a tile at a time,
    // then fill out along columns a row at a time
    if( colPeriod != -1 )
    {
        int nTiles = (mtx.GetNumColumns() - 2*haloWidth) / colPeriod;
        if( (mtx.GetNumColumns() - 2*haloWidth) % colPeriod != 0 )
        {
            nTiles += 1;
        }

        for( unsigned int t = 1; t < nTiles; t++ )
        {
            for( unsigned int i = 0; i < nTileRows; i++ )
            {
                memcpy( &(mtx.GetData()[haloWidth + i][haloWidth + t*nTileCols]),
                        &(mtx.GetData()[haloWidth + i][haloWidth]),
                        nTileCols * sizeof(T) );
            }
        }
    }
    if( rowPeriod != -1 )
    {
        int nTiles = (mtx.GetNumRows() - 2*haloWidth) / rowPeriod;
        if( (mtx.GetNumRows() - 2*haloWidth) % rowPeriod != 0 )
        {
            nTiles += 1;
        }

        for( unsigned int t = 1; t < nTiles; t++ )
        {
            for( unsigned int i = 0; i < nTileRows; i++ )
            {
                memcpy( &(mtx.GetData()[haloWidth + t*nTileRows + i][haloWidth]),
                        &(mtx.GetData()[haloWidth + i][haloWidth]),
                        (mtx.GetNumColumns() - 2*haloWidth) * sizeof(T) );
            }
        }
    }

    // initialize halo
    for( unsigned int i = 0; i < mtx.GetNumRows(); i++ )
    {
        for( unsigned int j = 0; j < mtx.GetNumColumns(); j++ )
        {
            bool inHalo = false;

            if( (i < haloWidth) || (i > mtx.GetNumRows() - 1 - haloWidth) )
            {
                inHalo = true;
            }
            else if( (j < haloWidth) || (j > mtx.GetNumColumns() - 1 - haloWidth) )
            {
                inHalo = true;
            }

            if( inHalo )
            {
                mtx.GetData()[i][j] = haloVal;
            }
        }
    }
}

