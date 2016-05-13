#ifndef CUDASTENCILFACTORY_H
#define CUDASTENCILFACTORY_H

#include "CommonCUDAStencilFactory.h"

template<class T>
class CUDAStencilFactory : public CommonCUDAStencilFactory<T>
{
public:
    CUDAStencilFactory( void )
      : CommonCUDAStencilFactory<T>( "CUDAStencil" )
    {
        // nothing else to do
    }

    virtual Stencil<T>* BuildStencil( const OptionParser& opts );
};

#endif // CUDASTENCILFACTORY_H

