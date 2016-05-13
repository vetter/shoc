#include <iostream>
#include <string>
#include <cassert>
#include "CUDAStencilFactory.h"
#include "CUDAStencil.h"


template<class T>
Stencil<T>*
CUDAStencilFactory<T>::BuildStencil( const OptionParser& options )
{
    // get options for base class
    T wCenter;
    T wCardinal;
    T wDiagonal;
    size_t lRows;
    size_t lCols;
    std::vector<long long int> devs;
    this->ExtractOptions( options,
                          wCenter,
                          wCardinal,
                          wDiagonal,
                          lRows,
                          lCols,
                          devs );

    // determine whcih device to use
    // We would really prefer this to be done in main() but
    // since BuildStencil is a virtual function, we cannot change its
    // signature, and OptionParser provides no way to override an
    // options' value after it is set during parsing.
    int chosenDevice = (int)devs[0];

    return new CUDAStencil<T>( wCenter,
                                wCardinal,
                                wDiagonal,
                                lRows,
                                lCols,
                                chosenDevice );
}


