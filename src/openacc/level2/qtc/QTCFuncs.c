#include "QTCFuncs.h"
#include "CTimer.h"


void
DoFloatQTC( float* points,
                    unsigned int numPoints,
                    float threshold,
                    double* clusteringTime,
                    double* totalTime )
{
    int numClusters = 0;

    int totalTimerHandle = Timer_Start();


    unsigned int* clusterMap = (unsigned int*)malloc(numPoints * sizeof(unsigned int));
    unsigned int* pointMap = (unsigned int*)malloc(numPoints * sizeof(unsigned int));

    int clusteringTimeHandle = Timer_Start();
    numClusters = QT2( points, 
                        numPoints, 
                        threshold, 
                        clusterMap, 
                        0,          // numClusters on first call
                        pointMap );

    *clusteringTime = Timer_Stop( clusteringTimerHandle, "" );

    free( clusterMap );
    free( pointMap );

    *totalTime = Timer_Stop( totalTimerHandle, "" );
}


int
QT2( float* points,
        unsigned int numPoints,
        float threshold,
        unsigned int* clusterMap,
        unsigned int numClusters,
        unsigned int* pointMap )
{
    unsigned int clusterSizes = (unsigned int*)malloc(numPoints * sizeof(unsigned int));

    unsigned int maxClusterSize = 0;
    unsigned int maxClusterIdx = 0;

#pragma omp parallel
    {
        int* local_isPointInCluster = (int*)malloc( numPoints * sizeof(int) );
        float* local_work = (float*)malloc( numPoints * 2 * sizeof(float) );
        int myid = omp_get_thread_num();

        // For each point, find the largest cluster we can when using that
        // point as a seed.
#pragma omp for
        for( unsigned int i = 0; i < numPoints; i++ )
        {
            clusterSizes[i] = FindCluster(points, i, local_isPointInCluster, local_work, numPoints, threshold);
        }
        free( local_isPointInCluster );
        free( local_work );

    }

#if READY
    add openacc pragma with max reduce on this reduce(max : maxClusterSize)
    then linear search for idx that gives that max value?
#endif // READY
        for( unsigned int i = 0; i < numPoints; i++ )
        {
            if( clusterSizes[i] > maxClusterSize )
            {
                maxClusterSize = clusterSizes[i];
                maxClusterIdx = i;
            }
        }

    int* isPointInCluster = (int*)malloc( numPoints * sizeof(int) );
    float* work = (float*)malloc( numPoints * 2 * sizeof(float) );
    unsigned int clusterSize = FindCluster(points,
                                            maxClusterIdx,
                                            isPointInCluster, 
                                            work, 
                                            numPoints, 
                                            threshold);

    // since the cluster we built was from the same seed point that
    // was previously shown to produce the maximum-sized cluster, it had
    // better be the same size as the maximum-sized cluster.
    if( clusterSize != maxClusterSize )
    {
        fprintf(stderr, "clusterSize=%d != maxClusterSize=%d\n", clusterSize, maxClusterSize);
    }
    assert( clusterSize == maxClusterSize );

    // Remove the cluster we just found from the set of points,
    // and cluster the rest.
    unsigned int newNumPoints = numPoints - clusterSize;
    unsigned int pointsHandled = 0;

    if( newNumPoints <= 1 )
    {
        for( unsigned int i = 0; i < numPoints; i++ )
        {
            if( !isPointInCluster[i] )
            {
                // the final cluster (if it exists) consists of one elt
                assert(++pointsHandled == 1);
                numClusters++;
            }
            clusterMap[pointMap[i]] = numClusters;
        }
    }
    else
    {
        assert(newNumPoints > 1);

        float* newPoints = (float*)malloc( newNumPoints * 2 * sizeof(float) );
        unsigned int* newPointMap = (unsigned int*)malloc( newNumPoints * sizeof(unsigned int));

        for( unsigned int i = 0; i < numPoints; i++ )
        {
            if( !isPointInCluster[i] )
            {
                newPointMap[pointsHandled] = pointMap[i];
                newPoints[2 * pointsHandled] = points[2 * i];
                newPoints[2 * pointsHandled + 1] = points[2 * i + 1];
                pointsHandled++;
            }
            else
            {
                clusterMap[pointMap[i]] = numClusters;
            }
        }
        // by now, we should have dealt with all points
        assert( pointsHandled == newNumPoints );

        numClusters = QT2( newPoints,
                            newNumPoints,
                            threshold,
                            clusterMap,
                            numClusters + 1,
                            newPointMap );

        free( newPoints );
        free( newPointMap );
    }

    return numClusters;
}


