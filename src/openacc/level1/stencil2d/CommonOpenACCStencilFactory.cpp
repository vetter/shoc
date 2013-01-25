#include <iostream>
#include <string>
#include <cassert>
#include "CommonOpenACCStencilFactory.h"
#include "InvalidArgValue.h"




template<class T>
void 
CommonOpenACCStencilFactory<T>::CheckOptions( const OptionParser& opts ) const
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

    // TODO any other tests we need to do on the custom size?
    // this is unlike OpenCL and CUDA, where the local dimensions
    // must evenly divide the global dimensions
}


template<class T>
void
CommonOpenACCStencilFactory<T>::ExtractOptions( const OptionParser& options,
                                            T& wCenter,
                                            T& wCardinal,
                                            T& wDiagonal )
{
    // let base class extract its options
    StencilFactory<T>::ExtractOptions( options, wCenter, wCardinal, wDiagonal );
    
    // nothing else to do, since we do not have local work group dimensions
}


