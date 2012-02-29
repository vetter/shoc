#ifndef OPENCLSTENCILFACTORY_H
#define OPENCLSTENCILFACTORY_H

#include "CommonOpenCLStencilFactory.h"

template<class T>
class OpenCLStencilFactory : public CommonOpenCLStencilFactory<T>
{
public:
    OpenCLStencilFactory( cl::Device& _dev,
                            cl::Context& _ctx,
                            cl::CommandQueue& _queue )
      : CommonOpenCLStencilFactory<T>( "OpenCLStencil", _dev, _ctx, _queue )
    {
        // nothing else to do
    }

    virtual Stencil<T>* BuildStencil( const OptionParser& options );
};

#endif // OPENCLSTENCILFACTORY_H
