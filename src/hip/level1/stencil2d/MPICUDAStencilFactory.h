#ifndef MPICUDASTENCILFACTORY_H
#define MPICUDASTENCILFACTORY_H

#include "CommonCUDAStencilFactory.h"

// ****************************************************************************
// Class:  MPICUDAStencilFactory
//
// Purpose:
//   MPI implementation of the CUDA stencil factory.
//
// Programmer:  Phil Roth
// Creation:    November 5, 2009
//
// ****************************************************************************
template<class T>
class MPICUDAStencilFactory : public CommonCUDAStencilFactory<T>
{
public:
    MPICUDAStencilFactory( void )
      : CommonCUDAStencilFactory<T>( "MPICUDAStencil" )
    {
        // nothing else to do
    }

    virtual Stencil<T>* BuildStencil( const OptionParser& options );
    virtual void CheckOptions( const OptionParser& options ) const;
};

#endif // MPICUDASTENCILFACTORY_H
