#include <iostream>
#include "HostStencilFactory.h"
#include "HostStencil.h"

template<class T>
Stencil<T>*
HostStencilFactory<T>::BuildStencil( const OptionParser& options )
{
    // get options for base class
    T wCenter;
    T wCardinal;
    T wDiagonal;
    StencilFactory<T>::ExtractOptions( options, wCenter, wCardinal, wDiagonal );

    return new HostStencil<T>( wCenter, wCardinal, wDiagonal );
}


template<class T>
void
HostStencilFactory<T>::CheckOptions( const OptionParser& options ) const
{
    // let base class check its options
    StencilFactory<T>::CheckOptions( options );

    // nothing else to do - we add no options
}


