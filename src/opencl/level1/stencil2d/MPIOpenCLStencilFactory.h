#ifndef MPIOPENCLSTENCILFACTORY_H
#define MPIOPENCLSTENCILFACTORY_H

#include "CommonOpenCLStencilFactory.h"

// ****************************************************************************
// Class:  MPIOpenCLStencilFactory
//
// Purpose:
//   MPI implementation of the OpenCL stencil factory.
//
// Programmer:  Phil Roth
// Creation:    November 5, 2009
//
// ****************************************************************************
template<class T>
class MPIOpenCLStencilFactory : public CommonOpenCLStencilFactory<T>
{
public:
    MPIOpenCLStencilFactory( cl_device_id _dev,
                                cl_context _ctx,
                                cl_command_queue _queue )
      : CommonOpenCLStencilFactory<T>( "MPIOpenCLStencil", _dev, _ctx, _queue )
    {
        // nothing else to do
    }

    virtual Stencil<T>* BuildStencil( const OptionParser& options );
    virtual void CheckOptions( const OptionParser& options ) const;
};

#endif // MPIOPENCLSTENCILFACTORY_H
