#ifndef QTCFUNCS_H
#define QTCFUNCS_H

void DoFloatQTC( float* points,
                    unsigned int numPoints,
                    float threshold,
                    double* clusteringTime,
                    double* totalTime )


#if READY
// if double precision is to be supported
void DoDoubleQTC( void );
#endif // READY

#endif // QTCFUNCS_H
