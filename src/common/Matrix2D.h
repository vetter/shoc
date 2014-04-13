#ifndef MATRIX2D_H
#define MATRIX2D_H

#include <iostream>
#ifdef _WIN32
#define restrict __restrict
#else
#include "config.h"
#endif

#include "PMSMemMgr.h"


// ****************************************************************************
// Class:  Matrix2D
//
// Purpose:
//   Encapsulation of 2D matrices.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
class Matrix2D
{
public:
    typedef T* restrict FlatDataPtr;
    typedef T* restrict* restrict DataPtr;
    typedef T* const restrict* const restrict ConstDataPtr;

private:
    static PMSMemMgr<T>* pmsmm;

    size_t nRows;
    size_t nColumns;
    size_t pad;
    size_t nPaddedColumns;
    FlatDataPtr flatData;   // 1D array of data
    DataPtr data;           // data as 2D array (ptr to array of ptrs)


    static size_t FindNumPaddedColumns( size_t nColumns, size_t pad )
    {
        return nColumns +
                 ((nColumns % pad == 0) ?
                    0 :
                    (pad - (nColumns % pad)));
    }

    void Init( void )
    {
        nPaddedColumns =  FindNumPaddedColumns( nColumns, pad );

        flatData = pmsmm->AllocHostBuffer( nRows * nPaddedColumns );
        data = new T*[nRows];

        for( size_t i = 0; i < nRows; i++ )
        {
            data[i] = &(flatData[i * nPaddedColumns]);
        }
    }

public:
    Matrix2D( size_t _nRows, size_t _nColumns, size_t _pad = 16 )
      : nRows( _nRows ),
        nColumns( _nColumns ),
        pad( _pad ),
        nPaddedColumns( 0 ),
        flatData( NULL ),
        data( NULL )
    {
        if( pmsmm == NULL )
        {
            pmsmm = new DefaultPMSMemMgr<T>;
        }
        Init();
    }

    ~Matrix2D( void )
    {
        delete[] data;
        data = NULL;

        pmsmm->ReleaseHostBuffer( flatData );
        flatData = NULL;
    }


    static void SetAllocator( PMSMemMgr<T>* _mgr )   { pmsmm = _mgr; }


    void Reset( size_t _nRows, size_t _nColumns )
    {
        if( (_nRows != nRows) || (_nColumns != nColumns) )
        {
            delete[] data;
            pmsmm->ReleaseHostBuffer( flatData );

            nRows = _nRows;
            nColumns = _nColumns;
            Init();
        }
    }

    DataPtr GetData( void )
    {
        return data;
    }

    ConstDataPtr GetConstData( void ) const
    {
        return data;
    }

    FlatDataPtr GetFlatData( void )
    {
        return flatData;
    }

    size_t GetNumRows( void ) const { return nRows; }
    size_t GetNumColumns( void ) const { return nColumns; }
    size_t GetNumPaddedColumns( void ) const    { return nPaddedColumns; }

    size_t GetDataSize( void ) const { return nRows * nPaddedColumns * sizeof(T); }

    size_t GetPad( void ) const { return pad; }

    bool ReadFrom( std::istream& s );
    bool WriteTo( std::ostream& s ) const;
};


template<class T>
std::ostream&
operator<<( std::ostream& s, const Matrix2D<T>& m )
{
    typename Matrix2D<T>::ConstDataPtr mdata = m.GetConstData();

    for( unsigned int i = 0; i < m.GetNumRows(); i++ )
    {
        for( unsigned int j = 0; j < m.GetNumColumns(); j++ )
        {
            if( j != 0 )
            {
                s << '\t';
            }
            s << mdata[i][j];
        }
        s << '\n';
    }
    return s;
}

#endif /* MATRIX2D_H */
