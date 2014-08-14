#ifndef QTCFUNCS_H
#define QTCFUNCS_H

// A Point in the 2D plane.
// Note: this needs to stay a POS so that it can be used from both 
// C and C++ code.
typedef struct Point
{
    float x;
    float y;
} Point;


#if defined(__cplusplus)
extern "C"
{
#endif // defined(__cplusplus)


void DoFloatQTC( float* pointsAsFloats,
                    unsigned int numPoints,
                    float threshold,
                    double* clusteringTime,
                    double* totalTime );


#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#endif /* QTCFUNCS_H */
