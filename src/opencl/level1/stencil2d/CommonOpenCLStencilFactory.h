#ifndef COMMONOPENCLSTENCILFACTORY_H
#define COMMONOPENCLSTENCILFACTORY_H

#include <vector>
#include "StencilFactory.h"
#include "support.h"

// ****************************************************************************
// Class:  CommonOpenCLStencilFactory
//
// Purpose:
//   OpenCL implementation of the stencil factory.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
class CommonOpenCLStencilFactory : public StencilFactory<T>
{
protected:
    cl::Device& dev;
    cl::Context& ctx;
    cl::CommandQueue& queue;

    void ExtractOptions( const OptionParser& options,
                            T& wCenter,
                            T& wCardinal,
                            T& wDiagonal,
                            size_t& lRows,
                            size_t& lCols );

public:
    CommonOpenCLStencilFactory( std::string _sname,
                                cl::Device& _dev,
                                cl::Context& _ctx,
                                cl::CommandQueue& _queue )
      : StencilFactory<T>( _sname ),
        dev( _dev ),
        ctx( _ctx ),
        queue( _queue )
    {
        // nothing else to do
    }

    virtual void CheckOptions( const OptionParser& options ) const;
};

#endif // COMMONOPENCLSTENCILFACTORY_H
