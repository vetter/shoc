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
    std::vector<long long> shDims = opts.getOptionVecInt( "lsize" );
    if( shDims.size() != 2 )
    {
        throw InvalidArgValue( "lsize must have two dimensions" );
    }
    if( (shDims[0] <= 0) || (shDims[1] <= 0) )
    {
        throw InvalidArgValue( "all lsize values must be positive" );
    }

    std::vector<long long> arrayDims = opts.getOptionVecInt( "customSize" );
    assert( arrayDims.size() == 2 );

    // If both of these are zero, we're using a non-custom size, skip this test
    if (arrayDims[0] == 0 && arrayDims[0] == 0)
    {
        return;
    }

    size_t gRows = (size_t)arrayDims[0];
    size_t gCols = (size_t)arrayDims[1];
    size_t lRows = (size_t)shDims[0];
    size_t lCols = (size_t)shDims[1];

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
                                            size_t& lRows,
                                            size_t& lCols,
                                            std::vector<long long>& devices )
{
    // let base class extract its options
    StencilFactory<T>::ExtractOptions( options, wCenter, wCardinal, wDiagonal );

    // extract our options
    std::vector<long long> ldims = options.getOptionVecInt( "lsize" );
    assert( ldims.size() == 2 );
    lRows = (size_t)ldims[0];
    lCols = (size_t)ldims[1];

    // determine which device to use
    // We would really prefer this to be done in main() but
    // since BuildStencil is a virtual function, we cannot change its
    // signature, and OptionParser provides no way to override an
    // option's value after it is set during parsing.
    devices = options.getOptionVecInt("device");
}


