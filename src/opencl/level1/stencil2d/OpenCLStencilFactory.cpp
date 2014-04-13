#include <iostream>
#include <string>
#include <cassert>
#include "OpenCLStencilFactory.h"
#include "OpenCLStencil.h"
#include "OpenCLDeviceInfo.h"



template<class T>
Stencil<T>*
OpenCLStencilFactory<T>::BuildStencil( const OptionParser& options )
{
    // get options for base class
    T wCenter;
    T wCardinal;
    T wDiagonal;
    size_t lRows;
    size_t lCols;
    this->ExtractOptions( options,
                          wCenter,
                          wCardinal,
                          wDiagonal,
                          lRows,
                          lCols );

    // build the stencil object
    return new OpenCLStencil<T>( wCenter,
                                wCardinal,
                                wDiagonal,
                                lRows,
                                lCols,
                                this->dev,
                                this->ctx,
                                this->queue );
}


