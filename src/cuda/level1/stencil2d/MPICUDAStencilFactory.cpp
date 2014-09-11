#include "mpi.h"
#include <iostream>
#include <string>
#include <assert.h>
#include "NodeInfo.h"
#include "MPICUDAStencilFactory.h"
#include "MPICUDAStencil.h"
#include "InvalidArgValue.h"


template<class T>
Stencil<T>*
MPICUDAStencilFactory<T>::BuildStencil( const OptionParser& options )
{
    // get options for base class
    T wCenter;
    T wCardinal;
    T wDiagonal;
    size_t lRows;
    size_t lCols;
    std::vector<long long> devs;
    CommonCUDAStencilFactory<T>::ExtractOptions( options,
                                                wCenter,
                                                wCardinal,
                                                wDiagonal,
                                                lRows,
                                                lCols,
                                                devs );

    size_t mpiGridRows;
    size_t mpiGridCols;
    unsigned int nItersPerHaloExchange;
    MPI2DGridProgram<T>::ExtractOptions( options,
                                        mpiGridRows,
                                        mpiGridCols,
                                        nItersPerHaloExchange );

    // determine which device to use
    // We would really prefer this to be done in main() but
    // since BuildStencil is a virtual function, we cannot change its
    // signature, and OptionParser provides no way to override an
    // option's value after it is set during parsing.
    NodeInfo ni;
    int myRankInNode = ni.nodeRank();
    // assign tasks to devices in round-robin approach
    // TODO is this really the mapping we want if the CUDA-capable
    // devices are heterogeneous?
    int chosenDevice = (myRankInNode % devs.size());

    return new MPICUDAStencil<T>( wCenter,
                                wCardinal,
                                wDiagonal,
                                lRows,
                                lCols,
                                mpiGridRows,
                                mpiGridCols,
                                nItersPerHaloExchange,
                                chosenDevice
                                );
}


template<class T>
void
MPICUDAStencilFactory<T>::CheckOptions( const OptionParser& opts ) const
{
    // let base class check its options first
    CommonCUDAStencilFactory<T>::CheckOptions( opts );

    // check our options
    std::vector<long long> shDims = opts.getOptionVecInt( "lsize" );
    std::vector<long long> arrayDims = opts.getOptionVecInt( "customSize" );
    if( arrayDims[0] == 0 )
    {
        // custom size was not specified - we are using a standard size
        int sizeClass = opts.getOptionInt("size");
        arrayDims = StencilFactory<T>::GetStandardProblemSize( sizeClass );
    }
    assert( shDims.size() == 2 );
    assert( arrayDims.size() == 2 );

    size_t gRows = (size_t)arrayDims[0];
    size_t gCols = (size_t)arrayDims[1];
    size_t lRows = shDims[0];
    size_t lCols = shDims[1];

    unsigned int haloWidth = (unsigned int)opts.getOptionInt( "iters-per-exchange" );

    // verify that MPI halo width will result in a matrix being passed
    // to the kernel that also has its global size as a multiple of
    // the local work size
    //
    // Because the MPI halo width is arbitrary, and the kernel halo width
    // is always 1, we have to ensure that:
    //   ((size + 2*halo) - 2) % lsize == 0
    if( (((gRows + 2*haloWidth) - 2) % lRows) != 0 )
    {
        throw InvalidArgValue( "rows including halo must be even multiple of lsize (e.g., lsize rows evenly divides ((rows + 2*halo) - 2) )" );
    }
    if( (((gCols + 2*haloWidth) - 2) % lCols) != 0 )
    {
        throw InvalidArgValue( "columns including halo must be even multiple of lsize (e.g., lsize cols evenly divides ((cols + 2*halo) - 2) )" );
    }
}


