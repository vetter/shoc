#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif // HAVE_STDINT_H
#include "Matrix2D.h"

#ifdef _WIN32
typedef unsigned int uint32_t;
#endif


template<class T>
bool
Matrix2D<T>::ReadFrom( std::istream& s )
{
    uint32_t nRowsUint;
    uint32_t nColsUint;

    s.read( (char*)&nRowsUint, sizeof(nRowsUint) );
    s.read( (char*)&nColsUint, sizeof(nColsUint) );

    uint32_t nPaddedColsUint = FindNumPaddedColumns( nColsUint, pad );

    T* newDataFlat = new T[nRowsUint * nPaddedColsUint];
    T** newData = new T*[nRowsUint];
    for( size_t i = 0; i < nRowsUint; i++ )
    {
        newData[i] = &(newDataFlat[i * nPaddedColsUint]);
        s.read( (char*)newData[i], nColsUint * sizeof(T) );
    }

    if( s.good() )
    {
        // we successfully read the matrix
        // release any old data
        delete[] data;
        delete[] flatData;

        // re-initialize with new data
        nRows = nRowsUint;
        nColumns = nColsUint;
        nPaddedColumns = nPaddedColsUint;
        flatData = newDataFlat;
        data = newData;
    }
    else
    {
        delete[] newDataFlat;
        delete[] newData;
    }

    return s.good();
}


// note we do not write padding to output file
template<class T>
bool
Matrix2D<T>::WriteTo( std::ostream& s ) const
{
    uint32_t nRowsUint = nRows;
    uint32_t nColsUint = nColumns;

    s.write( (const char*)&nRowsUint, sizeof(nRowsUint) );
    s.write( (const char*)&nColsUint, sizeof(nColsUint) );
    for( uint32_t r = 0; r < nRows; r++ )
    {
        s.write( (const char*)data[r], nColumns * sizeof(T) );
    }

    return s.good();
}

