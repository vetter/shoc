#ifndef PARALLEL_TREE_MERGE_H
#define PARALLEL_TREE_MERGE_H

#include <mpi.h>
#include <string>
#include <stdlib.h>
#include "GetMPIType.h"

using namespace std;

// ****************************************************************************
// Class: ParallelTreeMerge
//
// Purpose:
//   Implements a generic tree-type reduction of an array of
//   elements of type T, in log(N) steps where N is the number of
//   processes participating in the reduction.
//
// Notes:     To use this class, a user must create a derived class
//   that implements the two pure virtual methods: getMergeData
//   and processMergeData.
//
// Programmer: Gabriel Marin
// Creation: August 25, 2009
//
// Modifications:
//
// ****************************************************************************

namespace SHOC {

const int MERGE_COUNT_TAG = 1;
const int MERGE_DATA_TAG = 2;

// define a base class for doing a tree based reduction in log(P) steps.
// This class implements a generic merge step that takes the data and
// calls a pre- and post- merge callback for further customization by
// derived classes. Template parameter defines the basic type (must be
// a type that has an MPI correspondent.
template <typename T>
class ParallelTreeMerge
{
protected:
    // getMergeData must be implmented by any class that extends
    // ParallelTreeMerge. getMergeData is invoked before each reduction
    // step for processes that must send data during that step.
    virtual const T* getMergeData (int *size, int _key = 0) = 0;

    // processMergeData must be implmented by any class that extends
    // ParallelTreeMerge. processMergeData is invoked after each reduction
    // step for processes that received data during that step.
    virtual void processMergeData (const T *_data, int size, int _key = 0) = 0;

public:

    // do a tree type reduction over nprocs, using communicator comm.
    // myrank is my rank.
    // key is a parameter that will be passed to the callback functions
    // so a sub-class can call the doMerge method for multiple arrays
    // and differentiate between them inside the callbacks based on key.
    void doMerge (int myrank, int nprocs, MPI_Comm comm, int key = 0)
    {
        int otherrank;
        int step = 0;
        int stepShift = 1;
        int receiver = 1;
        int mesgSize;
        T value;
        MPI_Status status;
        MPI_Datatype t = GetMPIType (value);

        do
        {
            otherrank = myrank ^ stepShift;
            receiver = (myrank < otherrank);

            if (otherrank < nprocs)   // I have a partner in this step
            {
                if (! receiver)   // sender
                {
                    const T* data = getMergeData(&mesgSize, key);
                    // send counts to the other rank
                    MPI_Send (&mesgSize, 1, MPI_INT, otherrank, MERGE_COUNT_TAG, comm);

                    MPI_Send ((void*)data, mesgSize, t, otherrank, MERGE_DATA_TAG, comm);
                } else    // receiver
                {
                    // get the size of the buffer to be received
                    MPI_Recv (&mesgSize, 1, MPI_INT, otherrank, MERGE_COUNT_TAG, comm, &status);
                    T *recvData = new T[mesgSize];
                    if (recvData == 0) {
                        cerr << "Task " << myrank
                             << " cannot allocate buffer for receiving data during merge step "
                             << step << "." << endl;
                        exit (-3);
                    }

                    // receive data from the other rank
                    MPI_Recv (recvData, mesgSize, t, otherrank, MERGE_DATA_TAG, comm, &status);
                    int recvLen = mesgSize;
                    MPI_Get_count(&status, t, &recvLen);  // see exactly how much we received
                    processMergeData (recvData, recvLen, key);
                }
            }

            step += 1;
            stepShift <<= 1;
        } while (stepShift<nprocs && receiver);
    }
};

};

#endif
