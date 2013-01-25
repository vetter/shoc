#ifndef COMMONOPENACCSTENCILFACTORY_H
#define COMMONOPENACCSTENCILFACTORY_H

#include "StencilFactory.h"

// ****************************************************************************
// Class:  CommonOpenACCStencilFactory
//
// Purpose:
//   OpenACC implementation of the stencil factory.
//
// Programmer:  Phil Roth
// Creation:    2012-12-06
//
// ****************************************************************************
template<class T>
class CommonOpenACCStencilFactory : public StencilFactory<T>
{
protected:
    void ExtractOptions( const OptionParser& options,
                            T& wCenter,
                            T& wCardinal,
                            T& wDiagonal );

public:
    CommonOpenACCStencilFactory( std::string _sname )
      : StencilFactory<T>( _sname )
    {
        // nothing else to do
    }

    virtual void CheckOptions( const OptionParser& options ) const;
};

#endif // COMMONOPENACCSTENCILFACTORY_H
