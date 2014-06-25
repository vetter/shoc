#ifndef MPI2DGRIDPROGRAM_H
#define MPI2DGRIDPROGRAM_H

#include "mpi.h"
#include "Matrix2D.h"
#include "OptionParser.h"

// ****************************************************************************
// Class:  MPI2DGridProgramBase
//
// Purpose:
//   Encapsulation of a program which operates on a 2D grid and uses MPI
//   to divide work among tasks.
//
// Programmer:  Phil Roth
// Creation:    November 5, 2009
//
// ****************************************************************************
class MPI2DGridProgramBase
{
protected:
    enum
    {
        haloExchangeToNorth = 7,
        haloExchangeToSouth = 8,
        haloExchangeToEast = 9,
        haloExchangeToWest = 10
    };

    int cwrank;         // rank in MPI_COMM_WORLD
    int cwsize;         // size of MPI_COMM_WORLD
    MPI_Comm comm2d;    // communicator with 2D cartesian topology
    int dims[2];        // dimensions of 2D cartesian topology
    int coords[2];      // coordinates in 2D cartesian topology
    unsigned int haloWidth;


    unsigned int GetNumberIterationsPerHaloExchange( void ) const
        {
            return haloWidth;
        }

    bool HaveNorthNeighbor( void ) const
        {
            return (coords[0] > 0);
        }

    bool HaveSouthNeighbor( void ) const
        {
            return ( coords[0] < (dims[0] - 1) );
        }

    bool HaveWestNeighbor( void ) const
        {
            return ( coords[1] > 0 );
        }

    bool HaveEastNeighbor( void ) const
        {
            return ( coords[1] < (dims[1] - 1) );
        }

    int GetCommWorldRank( void ) const  { return cwrank; }
    int GetCommWorldSize( void ) const  { return cwsize; }

    MPI2DGridProgramBase( size_t mpiGridRows,
                        size_t mpiGridCols,
                        unsigned int nItersPerHaloExchange );

public:
    static void AddOptions( OptionParser& opts );
    static void CheckOptions( const OptionParser& opts );
    static void ExtractOptions( const OptionParser& opts,
                            size_t& mpiGridRows,
                            size_t& mpiGridCols,
                            unsigned int& nItersPerHaloExchange );

    virtual ~MPI2DGridProgramBase( void )
    {
        // nothing else to do
    }

    bool ParticipatingInProgram( void ) const   { return (coords[0] != -1); }
};


// ****************************************************************************
// Class:  MPI2DGridProgram
//
// Purpose:
//   An MPI2DGridProgram that knows how to work with matrices of a specific
//   type T.
//
// Programmer:  Phil Roth
// Creation:    Feb 29, 2012
//
// ****************************************************************************
template<class T>
class MPI2DGridProgram : public MPI2DGridProgramBase
{
private:
    MPI_Datatype GetMPIDataType( void ) const;

protected:
    void DoHaloExchange( Matrix2D<T>& mtx );
    void DumpData( std::ostream& s,
                    const Matrix2D<T>& adata,
                    const char* desc );

public:
    MPI2DGridProgram( size_t _mpiGridRows,
                        size_t _mpiGridCols,
                        unsigned int _nItersPerHaloExchange )
      : MPI2DGridProgramBase( _mpiGridRows,
                                _mpiGridCols,
                                _nItersPerHaloExchange )
    {
        // nothing else to do
    }

    virtual ~MPI2DGridProgram( void )
    {
        // nothing else to do
    }
};

#endif /* MPI2DGRIDPROGRAM_H */
