#ifndef HOSTSTENCILFACTORY_H
#define HOSTSTENCILFACTORY_H

#include "StencilFactory.h"

// ****************************************************************************
// Class:  HostStencilFactory
//
// Purpose:
//   Class to generate stencils for hosts.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
class HostStencilFactory: public StencilFactory<T>
{
public:
    HostStencilFactory( void )
      : StencilFactory<T>( "HostStencil" )
    {
        // nothing else to do
    }

    virtual Stencil<T>* BuildStencil( const OptionParser& options );
    virtual void CheckOptions( const OptionParser& options ) const;
};

#endif // HOSTSTENCILFACTORY_H
