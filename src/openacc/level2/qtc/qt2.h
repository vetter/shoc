#ifndef QT2_H
#define QT2_H

#include <omp.h>
#define MAX_THREADS 64

#define PAD_SIZE    (64 - (2*sizeof(unsigned int)+2*sizeof(int*)+sizeof(float*)))
struct local_data_t
{
    unsigned int maxClusterSize;
    unsigned int maxClusterIdx;
    int* ptIsInCluster;
    int* work;
    float* maxDist;
    char pad[PAD_SIZE];
};
extern local_data_t local[];

#endif // QT2_H
