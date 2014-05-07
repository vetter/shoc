#include "mpi.h"
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cassert>
#include "MPIHostStencil.h"


template<class T>
MPIHostStencil<T>::MPIHostStencil( T _wCenter,
                                T _wCardinal,
                                T _wDiagonal,
                                size_t _mpiGridRows,
                                size_t _mpiGridCols,
                                unsigned int _nItersPerHaloExchange,
                                bool _dumpData )
  : HostStencil<T>( _wCenter,
                _wCardinal,
                _wDiagonal ),
    MPI2DGridProgram<T>( _mpiGridRows,
                _mpiGridCols,
                _nItersPerHaloExchange ),
    dumpData( _dumpData )
{
    if( dumpData )
    {
        std::ostringstream fnamestr;
        fnamestr << "host." << std::setw( 4 ) << std::setfill('0') << this->GetCommWorldRank();
        ofs.open( fnamestr.str().c_str() );
    }
}


template<class T>
void
MPIHostStencil<T>::operator()( Matrix2D<T>& mtx, unsigned int nIters )
{
    if( this->ParticipatingInProgram() )
    {
        HostStencil<T>::operator()( mtx, nIters );
        if( dumpData )
        {
            this->DumpData( ofs, mtx, "after all iterations" );
        }
    }
    MPI_Barrier( MPI_COMM_WORLD );
}


template<class T>
void
MPIHostStencil<T>::DoPreIterationWork( Matrix2D<T>& mtx, unsigned int iter )
{
    if( (iter % this->GetNumberIterationsPerHaloExchange() ) == 0 )
    {
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
}


