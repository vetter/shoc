#include "mpi.h"
#include <iostream>
#include "MPIHostStencilFactory.h"
#include "MPIHostStencil.h"

template<class T>
Stencil<T>*
MPIHostStencilFactory<T>::BuildStencil( const OptionParser& opts )
{
    // get options for base class
    T wCenter;
    T wCardinal;
    T wDiagonal;
    StencilFactory<T>::ExtractOptions( opts, wCenter, wCardinal, wDiagonal );

    // get our options
    std::vector<long long> mpiDims = opts.getOptionVecInt( "msize" );
    long nItersPerExchange = opts.getOptionInt( "iters-per-exchange" );

    return new MPIHostStencil<T>( wCenter,
                                wCardinal,
                                wDiagonal,
                                (size_t)mpiDims[0],
                                (size_t)mpiDims[1],
                                (unsigned int)nItersPerExchange
                                );
}


template<class T>
void
MPIHostStencilFactory<T>::AddOptions( OptionParser& opts ) const
{
    MPI2DGridProgram<T>::AddOptions( opts );
}


template<class T>
void
MPIHostStencilFactory<T>::CheckOptions( const OptionParser& opts ) const
{
    // let base class check its options
    StencilFactory<T>::CheckOptions( opts );

    // check our options
    MPI2DGridProgram<T>::CheckOptions( opts );
}


