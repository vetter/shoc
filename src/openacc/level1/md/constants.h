#ifndef CONSTANTS_H__
#define CONSTANTS_H__

// Problem Constants
static const float  cutsq        = 16.0f; // Square of cutoff distance
static const int    maxNeighbors = 128;   // Max number of nearest neighbors
static const double domainEdge   = 20.0;  // Edge length of the cubic domain
static const float  lj1          = 1.5;   // LJ constants
static const float  lj2          = 2.0;
static const float  EPSILON      = 0.1f;  // Relative Error between CPU/GPU

// Float vector types
typedef struct {
    float x;
    float y;
    float z;
} float3;

typedef struct {
    float x;
    float y;
    float z;
    float w;
} float4;

typedef struct {
    double x;
    double y;
    double z;
} double3;

typedef struct {
    double x;
    double y;
    double z;
    double w;
} double4;


#endif // CONSTANTS_H__
