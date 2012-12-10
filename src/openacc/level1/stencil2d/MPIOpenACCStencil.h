#ifndef MPIOPENACCSTENCIL_H
#define MPIOPENACCSTENCIL_H

#include <fstream>
#include <vector>
#include "OpenACCStencil.h"
#include "MPI2DGridProgram.h"


// ****************************************************************************
// Class:  MPIOpenACCStencil
//
// Purpose:
//   MPI implementation of OpenACC stencil
//
// Programmer:  Phil Roth
// Creation:    2012-12-10
//
// ****************************************************************************
template<class T>
class MPIOpenACCStencil : public OpenACCStencil<T>, public MPI2DGridProgram<T>
{
private:
    std::ofstream ofs;
    bool dumpData;

    T* eData;
    T* wData;

    void DoPreIterBlockWork( Matrix2D<T>& mtx );

    static void PreIterBlockFuncCB( void* cbData );

public:
    MPIOpenACCStencil( T wCenter,
                    T wCardinal,
                    T wDiagonal,
                    size_t _mpiGridRows,
                    size_t _mpiGridCols,
                    unsigned int _nItersPerHaloExchange,
                    bool _dumpData = false );
    virtual ~MPIOpenACCStencil( void );

    virtual void operator()( Matrix2D<T>&, unsigned int nIters );
};

#endif // MPIOPENACCSTENCIL_H
