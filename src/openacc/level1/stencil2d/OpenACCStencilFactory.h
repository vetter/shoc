#ifndef OPENACCSTENCILFACTORY_H
#define OPENACCSTENCILFACTORY_H

#include "CommonOpenACCStencilFactory.h"

template<class T>
class OpenACCStencilFactory : public CommonOpenACCStencilFactory<T>
{
public:
    OpenACCStencilFactory( void )
      : CommonOpenACCStencilFactory<T>( "OpenACCStencil" )
    {
        // nothing else to do
    }

    virtual Stencil<T>* BuildStencil( const OptionParser& options );
};

#endif // OPENACCSTENCILFACTORY_H
