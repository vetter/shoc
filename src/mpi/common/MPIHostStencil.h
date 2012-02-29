#ifndef MPIHOSTSTENCIL_H
#define MPIHOSTSTENCIL_H

#include <fstream>
#include "mpi.h"
#include "HostStencil.h"
#include "MPI2DGridProgram.h"


// ****************************************************************************
// Class:  MPIHostStencil
//
// Purpose:
//   Stencils for MPI hosts.
//
// Programmer:  Phil Roth
// Creation:    November 5, 2009
//
// ****************************************************************************
template<class T>
class MPIHostStencil : public HostStencil<T>, public MPI2DGridProgram<T>
{
private:
    std::ofstream ofs;
    bool dumpData;

protected:
    virtual void DoPreIterationWork( Matrix2D<T>& mtx, unsigned int iter );

public:
    MPIHostStencil( T wCenter,
                        T wCardinal,
                        T wDiagonal,
                        size_t mpiGridRows,
                        size_t mpiGridCols,
                        unsigned int nItersPerHaloExchange,
                        bool dumpData = false );

    virtual void operator()( Matrix2D<T>&, unsigned int nIters );
};

#endif /* MPIHOSTSTENCIL_H */
