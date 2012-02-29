#ifndef MPIHOSTSTENCILFACTORY_H
#define MPIHOSTSTENCILFACTORY_H

#include "StencilFactory.h"

// ****************************************************************************
// Class:  MPIHostStencilFactory
//
// Purpose:
//   Class to generate stencils for MPI Hosts.
//
// Programmer:  Phil Roth
// Creation:    November 5, 2009
//
// ****************************************************************************
template<class T>
class MPIHostStencilFactory: public StencilFactory<T>
{
public:
    MPIHostStencilFactory( void )
      : StencilFactory<T>( "MPIHostStencil" )
    {
        // nothing else to do
    }

    virtual Stencil<T>* BuildStencil( const OptionParser& options );
    virtual void AddOptions( OptionParser& odesc ) const;
    virtual void CheckOptions( const OptionParser& options ) const;
};

#endif // MPIHOSTSTENCILFACTORY_H
