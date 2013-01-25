#include <iostream>
#include <string>
#include <cassert>
#include "OpenACCStencilFactory.h"
#include "OpenACCStencil.h"



template<class T>
Stencil<T>*
OpenACCStencilFactory<T>::BuildStencil( const OptionParser& options )
{
    // get options for base class
    T wCenter;
    T wCardinal;
    T wDiagonal;
    ExtractOptions( options,
                    wCenter,
                    wCardinal,
                    wDiagonal );

    // build the stencil object
    return new OpenACCStencil<T>( wCenter, 
                                wCardinal, 
                                wDiagonal );
}


