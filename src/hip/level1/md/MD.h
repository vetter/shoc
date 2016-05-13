#ifndef MD_H__
#define MD_H__

#include <cuda.h>
#include <cuda_runtime_api.h>

// Problem Constants
static const float  cutsq        = 16.0f; // Square of cutoff distance
static const int    maxNeighbors = 128;  // Max number of nearest neighbors
static const double domainEdge   = 20.0; // Edge length of the cubic domain
static const float  lj1          = 1.5;  // LJ constants
static const float  lj2          = 2.0;
static const float  EPSILON      = 0.1f; // Relative Error between CPU/GPU

#endif // MD_H__
