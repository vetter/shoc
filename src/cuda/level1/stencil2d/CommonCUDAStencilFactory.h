#ifndef COMMONCUDASTENCILFACTORY_H
#define COMMONCUDASTENCILFACTORY_H

#include <vector>
#include "StencilFactory.h"

// ****************************************************************************
// Class:  CommonCUDAStencilFactory
//
// Purpose:
//   CUDA implementation of stencil factory.
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
template<class T>
class CommonCUDAStencilFactory : public StencilFactory<T>
{
protected:
    void ExtractOptions( const OptionParser& options,
                            T& wCenter,
                            T& wCardinal,
                            T& wDiagonal,
                            size_t& lRows,
                            size_t& lCols,
                            std::vector<long long>& devices );

public:
    CommonCUDAStencilFactory( std::string _sname )
      : StencilFactory<T>( _sname )
    {
        // nothing else to do
    }

    virtual void CheckOptions( const OptionParser& opts ) const;
};

#endif // COMMONCUDASTENCILFACTORY_H

