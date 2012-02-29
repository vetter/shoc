#include <iostream>
#include <string>
#include <cassert>
#include "CommonCUDAStencilFactory.h"
#include "InvalidArgValue.h"


template<class T>
void 
CommonCUDAStencilFactory<T>::CheckOptions( const OptionParser& opts ) const
{
    // let base class check its options first
    StencilFactory<T>::CheckOptions( opts );

    // check our options
    std::vector<long long> arrayDims = opts.getOptionVecInt( "customSize" );
    assert( arrayDims.size() == 2 );

    // If both of these are zero, we're using a non-custom size, skip this test
    if (arrayDims[0] == 0 && arrayDims[0] == 0)
    {
        return;
    }

    size_t gRows = (size_t)arrayDims[0];
    size_t gCols = (size_t)arrayDims[1];
    size_t lRows = LROWS;
    size_t lCols = LCOLS;

    // verify that local dimensions evenly divide global dimensions
    if( ((gRows % lRows) != 0) || (lRows > gRows) )
    {
        throw InvalidArgValue( "number of rows must be even multiple of lsize rows" );
    }
    if( ((gCols % lCols) != 0) || (lCols > gCols) )
    {
        throw InvalidArgValue( "number of columns must be even multiple of lsize columns" );
    }

    // TODO ensure local dims are smaller than CUDA implementation limits
}

template<class T>
void
CommonCUDAStencilFactory<T>::ExtractOptions( const OptionParser& options,
                                            T& wCenter,
                                            T& wCardinal,
                                            T& wDiagonal,
                                            std::vector<long long>& devices )
{
    // let base class extract its options
    StencilFactory<T>::ExtractOptions( options, wCenter, wCardinal, wDiagonal );

    // extract our options
    // with hardcoded lsize, we no longer have any to extract

    // determine which device to use
    // We would really prefer this to be done in main() but 
    // since BuildStencil is a virtual function, we cannot change its
    // signature, and OptionParser provides no way to override an
    // option's value after it is set during parsing.
    devices = options.getOptionVecInt("device");
}


