#ifndef RANDOM_PAIRS_H
#define RANDOM_PAIRS_H

// ****************************************************************************
// File: RandomPairs.h
//
// Purpose:
//   Collective method that pics a random, unique, pair each time called
//
// Programmer:  Vinod Tipparaju
// Creation:    August 12, 2009
//
// ****************************************************************************

int RandomPairs(int myrank, int numranks, MPI_Comm new_comm);

#endif
