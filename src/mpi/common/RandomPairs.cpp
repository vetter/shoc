#include "mpi.h"
#include <iostream>
#include <sstream>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NDEBUG

int mpi_error_code;
#define MP_ASSERT(error_code) if ((mpi_error_code = error_code) != MPI_SUCCESS)\
                                      MPI_Abort(MPI_COMM_WORLD, mpi_error_code)
// ****************************************************************************
// Function: RandomPairs
//
// Purpose:
//   Collective method that pics a random, unique, pair each time called
//
// Arguments:
//   arg1 : self rank
//   arg2 : total mpi ranks
//
// Returns:
//   the rank of the pair process
//
// Programmer: Vinod Tipparaju
// Creation:   August 12, 2009
//
// TODO:
//
// Modifications:
// 2010-02-24 - rothpc - converted to use permutation of ranks to assign pairs
//
// ****************************************************************************
int RandomPairs(int myrank, int numranks, MPI_Comm new_comm)
{
    int i;

    // initalize PRNG, use seed generated on processor 0 for uniform sequence
    int time_seed = time(NULL);
    MP_ASSERT(MPI_Bcast (&time_seed, 1, MPI_INT, 0, new_comm));
    srand(time_seed);
    rand();

    // build an array of task ranks
    int* ranks = new int[numranks];
    for( unsigned int i = 0; i < numranks; i++ )
    {
        ranks[i] = i;
    }

    // permute the array of task ranks
    // (we're using a Fisher-Yates shuffle, aka Knuth shuffle)
    for( i = (numranks - 1); i > 0; i-- )
    {
        // randomly select an element in [0,i]
        int ridx = (int)(i * ((double)(rand()) / RAND_MAX));
        assert( (ridx >= 0) && (ridx <= i) );

        // swap selected element with element at index i
        int tmp = ranks[ridx];
        ranks[ridx] = ranks[i];
        ranks[i] = tmp;
    }

    // determine my pair
    int mypair = -1;
    for( i = 0; i < numranks; i++ )
    {
        if( ranks[i] == myrank )
        {
            // we found my rank in the permutation
            // now check the position where we found it to see who my pair is
            if( ((numranks % 2) != 0) && (i == (numranks - 1)) )
            {
                // we have an odd number of ranks and I am the last one -
                // I am my own pair
                mypair = myrank;
            }
            else if( (i % 2) != 0 )
            {
                // My rank was found at an odd index -
                // my pair is the rank before mine
                mypair = ranks[i-1];
            }
            else
            {
                // My rank was found at an even index -
                // my pair is the rank after mine
                assert( (i % 2) == 0 );
                assert( (i+1) < numranks );
                mypair = ranks[i+1];
            }

            break;
        }
    }
    if( mypair == -1 )
    {
        std::ostringstream estr;
        estr << myrank << ": ranks=[";
        for( unsigned int i = 0; i < numranks; i++ )
        {
            estr << ranks[i] << " ";
        }
        estr << "]";
        std::cerr << estr.str() << std::endl;
    }
    assert( mypair != -1 );

    if( myrank == 0 )
    {
        std::ostringstream rstr;
        rstr << "RandomPair: pairs=";
        for( unsigned int i = 0; i < (numranks / 2); i++ )
        {
            rstr << ranks[2*i] << "->" << ranks[2*i+1] << " ";
        }

        if( (numranks % 2 ) != 0 )
        {
            rstr << ranks[numranks-1] << "->" << ranks[numranks-1] << " ";
        }
        std::cerr << rstr.str() << std::endl;
    }

    // clean up
    delete[] ranks;

    return mypair;
}

