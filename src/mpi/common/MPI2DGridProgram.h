#ifndef MPI2DGRIDPROGRAM_H
#define MPI2DGRIDPROGRAM_H

#include "mpi.h"
#include "Matrix2D.h"
#include "OptionParser.h"

// ****************************************************************************
// Class:  MPI2DGridProgram
//
// Purpose:
//   Encapsulation of a program which operates on a 2D grid and uses MPI
//   to divide work among tasks.
//
// Programmer:  Phil Roth
// Creation:    November 5, 2009
//
// ****************************************************************************
template<class T>
class MPI2DGridProgram
{
private:
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

    MPI_Datatype GetMPIDataType( void ) const;

protected:
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

    void DoHaloExchange( Matrix2D<T>& mtx );
    void DumpData( std::ostream& s,
                    const Matrix2D<T>& adata, 
                    const char* desc );
    int GetCommWorldRank( void ) const  { return cwrank; }
    int GetCommWorldSize( void ) const  { return cwsize; }

public:
    static void AddOptions( OptionParser& opts );
    static void CheckOptions( const OptionParser& opts );
    static void ExtractOptions( const OptionParser& opts,
                            size_t& mpiGridRows,
                            size_t& mpiGridCols,
                            unsigned int& nItersPerHaloExchange );

    MPI2DGridProgram( size_t mpiGridRows,
                        size_t mpiGridCols,
                        unsigned int nItersPerHaloExchange );
    virtual ~MPI2DGridProgram( void );

    bool ParticipatingInProgram( void ) const   { return (coords[0] != -1); }
};

#endif /* MPI2DGRIDPROGRAM_H */
