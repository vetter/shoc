#include "mpi.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include "MPI2DGridProgram.h"
#include "InvalidArgValue.h"

//----------------------------------------------------------------------------
// MPI2DGridProgramBase methods
//----------------------------------------------------------------------------

MPI2DGridProgramBase::MPI2DGridProgramBase( size_t _mpiGridRows,
                                    size_t _mpiGridCols,
                                    unsigned int _nItersPerHaloExchange )
  : haloWidth( _nItersPerHaloExchange )
{
    // find our place in the linear MPI topology
    MPI_Comm_size( MPI_COMM_WORLD, &cwsize );
    MPI_Comm_rank( MPI_COMM_WORLD, &cwrank );

    // build a communicator with 2D cartesian topology
    // for this test, we use non-periodic boundaries
    int nDims = 2;
    dims[0] = _mpiGridRows;
    dims[1] = _mpiGridCols;
    if( dims[0]*dims[1] > cwsize )
    {
        std::cerr << "2D mpi grid error: "
            << dims[0] << 'x' << dims[1]
            << " and " << cwsize << " total processes"
            << std::endl;
    }
    assert( dims[0]*dims[1] <= cwsize );

    if( cwrank == 0 )
    {
        if( dims[0]*dims[1] < cwsize )
        {
            std::cerr << "warning: using "
                << dims[0]*dims[1]
                << " of  " << cwsize
                << " available tasks; the rest will idle"
                << std::endl;
        }
    }

    int periodic[2];
    periodic[0] = 0;
    periodic[1] = 0;

    // Every process in MPI_COMM_WORLD must call MPI_Cart_create,
    // even if it won't be participating (because the 2D grid
    // that the user specified, or that we came up with by default,
    // does not require it).  Those that are not participating in
    // the computation will get MPI_COMM_NULL for comm2d.
    //
    MPI_Cart_create( MPI_COMM_WORLD,
                        2,          // number of dimensions
                        dims,
                        periodic,
                        1,          // reorder
                        &comm2d );
    if( comm2d != MPI_COMM_NULL )
    {
        // I am participating -
        // find our place in the 2D topology.
        MPI_Cart_coords( comm2d, cwrank, 2, coords );
    }
    else
    {
        // I am not participating in the computation.
        coords[0] = -1;
        coords[1] = -1;
    }
}


void
MPI2DGridProgramBase::AddOptions( OptionParser& opts )
{
    // rather than define a fixed 2D topology that will cause problems
    // if the user didn't specify exactly the right number of processes,
    // we build a topology based on the number of processes available.
    // For now, we are taking a simple approach and making it a 2D grid,
    // 2x(N/2) where N is the number of available processes.
    // At most, we idle one process using this approach but we use a true
    // 2D topology that forces halo exchanges in two or three directions
    // (as opposed to using a 1xN topology that would involve halo exchanges
    // across only one or two directions).
    //
    // A more sophisticated approach would determine a 2D topology that tries
    // to be as "square" as possible while keeping the number of processes
    // used <= the number of processes available.
    //
    int nprocs;
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    assert( nprocs > 0 );

    std::ostringstream ostr;
    if( nprocs > 1 )
    {
        int ny = (nprocs / 2);
        assert( ny > 0 );
        ostr << "2," << ny;
    }
    else
    {
        // We only have one process to work with -
        // our "2D" topology is a 1x1 degenerate grid.
        ostr << "1,1";
    }
    opts.addOption( "msize", OPT_VECINT, ostr.str().c_str(), "MPI 2D grid topology dimensions" );
    opts.addOption( "iters-per-exchange", OPT_INT, "1", "Number of local iterations between MPI boundary exchange operations (also, halo width)" );
}



void
MPI2DGridProgramBase::CheckOptions( const OptionParser& opts )
{
    std::vector<long long> mpiDims = opts.getOptionVecInt( "msize" );
    if( mpiDims.size() != 2 )
    {
        throw InvalidArgValue( "msize must be two-dimensional" );
    }
    if( (mpiDims[0] <= 0) || (mpiDims[1] <= 0) )
    {
        throw InvalidArgValue( "all msize dimensions must be positive" );
    }

    int cwsize;
    MPI_Comm_size( MPI_COMM_WORLD, &cwsize );
    if( mpiDims[0] * mpiDims[1] > cwsize )
    {
        throw InvalidArgValue( "msize dimensions specify more processes than are available" );
    }

    long nItersPerExchange = opts.getOptionInt( "iters-per-exchange" );
    if( nItersPerExchange <= 0 )
    {
        throw InvalidArgValue( "iterations per exchange must be positive" );
    }
}


void
MPI2DGridProgramBase::ExtractOptions( const OptionParser& opts,
                            size_t& mpiGridRows,
                            size_t& mpiGridCols,
                            unsigned int& nItersPerHaloExchange )
{
    std::vector<long long> mpiDims = opts.getOptionVecInt( "msize" );
    mpiGridRows = (size_t)mpiDims[0];
    mpiGridCols = (size_t)mpiDims[1];

    nItersPerHaloExchange = opts.getOptionInt( "iters-per-exchange" );
}


//----------------------------------------------------------------------------
// MPI2DGridProgram methods
//----------------------------------------------------------------------------
template<>
MPI_Datatype
MPI2DGridProgram<float>::GetMPIDataType( void ) const
{
    return MPI_FLOAT;
}


template<>
MPI_Datatype
MPI2DGridProgram<double>::GetMPIDataType( void ) const
{
    return MPI_DOUBLE;
}


template<class T>
void
MPI2DGridProgram<T>::DoHaloExchange( Matrix2D<T>& mtx )
{
    typename Matrix2D<T>::DataPtr adata = mtx.GetData();
    MPI_Datatype mpiElemDataType = GetMPIDataType();

    // exchange boundary values
    std::vector<MPI_Request> reqs;
    if( HaveNorthNeighbor() )
    {
        // I have a neighbor to the north
        int northNeighborRank = -1;
        int northNeighborCoords[2];
        northNeighborCoords[0] = coords[0] - 1;
        northNeighborCoords[1] = coords[1];
        MPI_Cart_rank( comm2d, northNeighborCoords, &northNeighborRank );

        // send and receive boundary data to that neighbor
        MPI_Request req;
        MPI_Irecv( &adata[0][0],
                    mtx.GetNumPaddedColumns() * haloWidth,
                    mpiElemDataType,
                    northNeighborRank,
                    haloExchangeToSouth,
                    MPI_COMM_WORLD,
                    &req );
        reqs.push_back( req );

        MPI_Send( &adata[haloWidth][0],
                    mtx.GetNumPaddedColumns() * haloWidth,
                    mpiElemDataType,
                    northNeighborRank,
                    haloExchangeToNorth,
                    MPI_COMM_WORLD );
    }
    if( HaveSouthNeighbor() )
    {
        // I have a neighbor to the south
        int southNeighborRank = -1;
        int southNeighborCoords[2];
        southNeighborCoords[0] = coords[0] + 1;
        southNeighborCoords[1] = coords[1];
        MPI_Cart_rank( comm2d, southNeighborCoords, &southNeighborRank );

        // send my own data to that neighbor
        MPI_Request req;
        MPI_Irecv( &adata[mtx.GetNumRows() - 1*haloWidth][0],
                    mtx.GetNumPaddedColumns() * haloWidth,
                    mpiElemDataType,
                    southNeighborRank,
                    haloExchangeToNorth,
                    MPI_COMM_WORLD,
                    &req );
        reqs.push_back( req );

        MPI_Send( &adata[mtx.GetNumRows() - 2*haloWidth][0],
                    mtx.GetNumPaddedColumns() * haloWidth,
                    mpiElemDataType,
                    southNeighborRank,
                    haloExchangeToSouth,
                    MPI_COMM_WORLD );
    }
    // wait for all north-south receives to finish
    MPI_Request* creqs = new MPI_Request[reqs.size()];
    std::copy( reqs.begin(), reqs.end(), creqs );
    MPI_Waitall( reqs.size(), creqs, MPI_STATUS_IGNORE );
    delete[] creqs;
    creqs = NULL;
    reqs.clear();

    // our dim1 data is not contiguous, so we can't do the simple send/receive
    // we used for north/south directions.
    // packing and unpacking into a contiguous buffer might be a bit faster than
    // using a derived datatype but makes our logic for handling the asynchronous
    // receives more difficult (because we'd have to unpack the east/west values
    // after we test that we've received them, and thus keep track of which
    // requests were for east boundary, and which were for the west boundary)
    MPI_Datatype dim1SliceType;
    MPI_Type_vector( mtx.GetNumRows(),   // number of blocks
                    haloWidth,           // number of elements in each block
                    mtx.GetNumPaddedColumns(),    // stride, in terms of number of elements
                    mpiElemDataType,           // old datatype
                    &dim1SliceType );    // new datatype
    MPI_Type_commit( &dim1SliceType );
    if( HaveWestNeighbor() )
    {
        // I have a neighbor to the west
        int westNeighborRank = -1;
        int westNeighborCoords[2];
        westNeighborCoords[0] = coords[0];
        westNeighborCoords[1] = coords[1] - 1;
        MPI_Cart_rank( comm2d, westNeighborCoords, &westNeighborRank );

        // send my own data to that neighbor

        MPI_Request req;
        MPI_Irecv( &adata[0][0],
                    1,
                    dim1SliceType,
                    westNeighborRank,
                    haloExchangeToEast,
                    MPI_COMM_WORLD,
                    &req );
        reqs.push_back( req );

        MPI_Send( &adata[0][1*haloWidth],
                    1,          // 1 vector, number of elements is encoded in type
                    dim1SliceType,
                    westNeighborRank,
                    haloExchangeToWest,
                    MPI_COMM_WORLD );
    }
    if( HaveEastNeighbor() )
    {
        // I have a neighbor to the east
        int eastNeighborRank = -1;
        int eastNeighborCoords[2];
        eastNeighborCoords[0] = coords[0];
        eastNeighborCoords[1] = coords[1] + 1;
        MPI_Cart_rank( comm2d, eastNeighborCoords, &eastNeighborRank );

        // send my own data to that neighbor

        MPI_Request req;
        MPI_Irecv( &adata[0][mtx.GetNumColumns() - 1*haloWidth],
                    1,
                    dim1SliceType,
                    eastNeighborRank,
                    haloExchangeToWest,
                    MPI_COMM_WORLD,
                    &req );
        reqs.push_back( req );

        MPI_Send( &adata[0][mtx.GetNumColumns() - 2*haloWidth],
                    1,          // 1 vector, number of elements is encoded in type
                    dim1SliceType,
                    eastNeighborRank,
                    haloExchangeToEast,
                    MPI_COMM_WORLD );
    }

    // wait for all east-west receives to finish
    creqs = new MPI_Request[reqs.size()];
    std::copy( reqs.begin(), reqs.end(), creqs );
    MPI_Waitall( reqs.size(), creqs, MPI_STATUS_IGNORE );
    delete[] creqs;
    creqs = NULL;

    // since we did the NS exchange first, then the EW exchange,
    // we swapped halo "corners"
}


template<class T>
void
MPI2DGridProgram<T>::DumpData( std::ostream& s,
                            const Matrix2D<T>& mtx,
                            const char* desc )
{
    typename Matrix2D<T>::ConstDataPtr adata = mtx.GetConstData();

    // dump our local data
    s << desc << std::endl;
    for( unsigned int i = 0; i < mtx.GetNumRows(); i++ )
    {
        for( unsigned int j = 0; j < mtx.GetNumColumns(); j++ )
        {
            s << std::setw(8) << adata[i][j] << ' ';
        }
        s << '\n';
    }
    s << std::endl;
}

