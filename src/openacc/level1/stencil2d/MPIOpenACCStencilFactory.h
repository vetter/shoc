#ifndef MPIOPENACCSTENCILFACTORY_H
#define MPIOPENACCSTENCILFACTORY_H

#include "CommonOpenACCStencilFactory.h"

// ****************************************************************************
// Class:  MPIOpenACCStencilFactory
//
// Purpose:
//   MPI implementation of the OpenACC stencil factory.
//
// Programmer:  Phil Roth
// Creation:    December 10, 2012
//
// ****************************************************************************
template<class T>
class MPIOpenACCStencilFactory : public CommonOpenACCStencilFactory<T>
{
public:
    MPIOpenACCStencilFactory( void )
      : CommonOpenACCStencilFactory<T>( "MPIOpenACCStencil" )
    {
        // nothing else to do
    }

    virtual Stencil<T>* BuildStencil( const OptionParser& options );
    virtual void CheckOptions( const OptionParser& options ) const;
};

#endif // MPIOPENACCSTENCILFACTORY_H
