#ifndef QTCFUNCS_H
#define QTCFUNCS_H

#if defined(__cplusplus)
extern "C"
{
#endif // defined(__cplusplus)

void DoFloatQTC( float* points,
                    unsigned int numPoints,
                    float threshold,
                    double* clusteringTime,
                    double* totalTime );


#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#endif /* QTCFUNCS_H */
