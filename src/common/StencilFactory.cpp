#include <assert.h>
#include "StencilFactory.h"
#include "InvalidArgValue.h"


template<class T>
void
StencilFactory<T>::CheckOptions( const OptionParser& options ) const
{
    // number of iterations must be positive
    unsigned int nIters = (unsigned int)options.getOptionInt( "num-iters" );
    if( nIters == 0 )
    {
        throw InvalidArgValue( "number of iterations must be positive" );
    }

    // no restrictions on weight values, just that we have them
}

template<class T>
void
StencilFactory<T>::ExtractOptions( const OptionParser& options,
                                T& wCenter,
                                T& wCardinal,
                                T& wDiagonal )
{
    wCenter = options.getOptionFloat( "weight-center" );
    wCardinal = options.getOptionFloat( "weight-cardinal" );
    wDiagonal = options.getOptionFloat( "weight-diagonal" );
}


template<class T>
std::vector<long long>
StencilFactory<T>::GetStandardProblemSize( int sizeClass )
{
    const int probSizes[4] = { 512, 1024, 2048, 4096 };
    if (!(sizeClass >= 0 && sizeClass < 5))
    {
        throw InvalidArgValue( "Size class must be between 1-4" );
    }

    std::vector<long long> ret( 2, probSizes[sizeClass - 1] );
    return ret;
}

