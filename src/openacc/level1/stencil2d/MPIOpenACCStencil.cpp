#include "mpi.h"
#include <sstream>
#include <iomanip>
#include <cassert>
#include "MPIOpenACCStencil.h"


// data to be passed to our PreIterBlockFuncData function
template<class T>
struct PreIterBlockFuncData
{
    MPIOpenACCStencil<T>* obj;
    Matrix2D<T>& mtx;

    PreIterBlockFuncData( MPIOpenACCStencil<T>* _obj, Matrix2D<T>& _mtx )
      : obj(_obj),
        mtx(_mtx)
    {
        // nothing else to do
    }
};


template<class T>
MPIOpenACCStencil<T>::MPIOpenACCStencil( T _wCenter,
                                    T _wCardinal,
                                    T _wDiagonal,
                                    size_t _mpiGridRows,
                                    size_t _mpiGridCols,
                                    unsigned int _nItersPerHaloExchange,
                                    bool _dumpData )
  : OpenACCStencil<T>( _wCenter,
                    _wCardinal,
                    _wDiagonal ),
    MPI2DGridProgram<T>( _mpiGridRows,
                    _mpiGridCols,
                    _nItersPerHaloExchange ),
    dumpData( _dumpData ),
    eData( NULL ),
    wData( NULL )
{
    if( dumpData )      
    {
        std::ostringstream fnamestr;
        fnamestr << "acc." << std::setw( 4 ) << std::setfill('0') << this->GetCommWorldRank();
        ofs.open( fnamestr.str().c_str() );
    }
}

template<class T>
MPIOpenACCStencil<T>::~MPIOpenACCStencil( void )
{
    delete[] eData;
    delete[] wData;
}


template<class T>
void 
MPIOpenACCStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    if( this->ParticipatingInProgram() )
    {
        // we need to do a halo exchange before our first push of
        // data onto the device
        if( dumpData )
        {
            this->DumpData( ofs, mtx, "before halo exchange" );
        }
        this->DoHaloExchange( mtx );
        if( dumpData )
        {
            this->DumpData( ofs, mtx, "after halo exchange" );
        }

        // allocate memory for halo exchange buffers for non-contiguous sides
        unsigned int haloWidth = this->GetNumberIterationsPerHaloExchange();
        size_t ewDataItemCount = haloWidth * mtx.GetNumRows();

        eData = new T[ewDataItemCount];
        wData = new T[ewDataItemCount];

        // apply the operator
        PreIterBlockFuncData<T> pibfd( this, mtx );
        this->ApplyOperator( mtx,
                            nIters,
                            this->GetNumberIterationsPerHaloExchange(),
                            PreIterBlockFuncCB,
                            &pibfd );

        if( dumpData )
        {
            this->DumpData( ofs, mtx, "after all iterations" );
        }
    }
    MPI_Barrier( MPI_COMM_WORLD );
}


template<class T>
void
MPIOpenACCStencil<T>::PreIterBlockFuncCB( void* cbData )
{
    PreIterBlockFuncData<T>* pibfd = static_cast<PreIterBlockFuncData<T>*>( cbData );
    assert( pibfd != NULL );
    assert( pibfd->obj != NULL );

    pibfd->obj->DoPreIterBlockWork( pibfd->mtx );
}




template<class T>
void
MPIOpenACCStencil<T>::DoPreIterBlockWork( Matrix2D<T>& mtx )
{
    // Unlike the CUDA and OpenCL versions of Stencil2D, 
    // this method is *not* called at the beginning of each iteration.
    // Instead, it is called only when we have to do a halo exchange.
    // We assume that the data we need to exchange is already in the given
    // "mtx" argument.

    if( dumpData )
    {
        this->DumpData( ofs, mtx, "before halo exchange" );
    }
    this->DoHaloExchange( mtx );
    if( dumpData )
    {
        this->DumpData( ofs, mtx, "after halo exchange" );
    }
}

