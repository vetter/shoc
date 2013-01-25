#include "mpi.h"
#include <iostream>
#include <string>
#include <cassert>
#include "MPIOpenACCStencilFactory.h"
#include "MPIOpenACCStencil.h"
#include "InvalidArgValue.h"


template<class T>
Stencil<T>*
MPIOpenACCStencilFactory<T>::BuildStencil( const OptionParser& options )
{
    // get options for base class
    T wCenter;
    T wCardinal;
    T wDiagonal;
    CommonOpenACCStencilFactory<T>::ExtractOptions( options,
                                                wCenter,
                                                wCardinal,
                                                wDiagonal );
    bool beVerbose = options.getOptionBool( "verbose" );

    size_t mpiGridRows;
    size_t mpiGridCols;
    unsigned int nItersPerHaloExchange;
    MPI2DGridProgram<T>::ExtractOptions( options,
                                        mpiGridRows,
                                        mpiGridCols,
                                        nItersPerHaloExchange );

    return new MPIOpenACCStencil<T>( wCenter,
                                wCardinal,
                                wDiagonal,
                                mpiGridRows,
                                mpiGridCols,
                                nItersPerHaloExchange,
                                beVerbose );
}


template<class T>
void
MPIOpenACCStencilFactory<T>::CheckOptions( const OptionParser& opts ) const
{
    // let base class check its options first
    CommonOpenACCStencilFactory<T>::CheckOptions( opts );

    // For OpenCL and CUDA, we have to check that the global dimensions
    // are a multiple of the local workgroup dimensions (plus taking 
    // the halo into effect).  For OpenACC, we don't (though we might
    // tend to achieve better performance if we did).
}

