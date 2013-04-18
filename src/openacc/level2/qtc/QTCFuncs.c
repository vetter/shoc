#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include "QTCFuncs.h"
#include "CTimer.h"

// Compute the Euclidean distance between two points.  
// NOTE: we return distance squared to avoid having to compute the
// square root.
inline
float
DistanceSquared( float* a, float* b )
{
    float dx = (a[0] - b[0]);
    float dy = (a[1] - b[1]);
    return (dx*dx + dy*dy);
}


// Find the diameter of the given cluster if the testPoint were added to it.
// Note: to avoid having to compute square roots, our distances are 
// "distance squared," so the diameter value reported by this function is 
// actually a distance squared.
float
Diameter(float* cluster, unsigned int size, float* testPoint)
{
    float maxDistanceSq = 0;
    for( unsigned int i = 0; i < size; i++ )
    {
        float currDistanceSq = DistanceSquared( &(cluster[i]), testPoint );
        if( currDistanceSq > maxDistanceSq )
        {
            maxDistanceSq = currDistanceSq;
        }
    }
    return maxDistanceSq;
}


unsigned int
FindCluster(float* points,
                unsigned int seedIdx, 
                int* isPointInCluster, 
                float* work, 
                unsigned int numPoints, 
                float threshold)
{
    // at the start, only the seed point is in the cluster
    unsigned int size = 1;
    memset(isPointInCluster, 0, numPoints*sizeof(int));
    isPointInCluster[seedIdx] = 1;

    work[0] = points[2*seedIdx];
    work[1] = points[2*seedIdx+1];

    while( size < numPoints )
    {
        float minDiameter = FLT_MAX;    // minimum diameter we've seen so far
        int minDiameterIdx;             // index of point that gave min diameter

        for( unsigned int j = 0; j < numPoints; j++ )
        {
            if( !isPointInCluster[j] )
            {
                // point is not already in the cluster
                // figure out what our diameter would be if we added it
                float testDiameter = Diameter(work, size, &(points[2*j]) );
                if( testDiameter < minDiameter )
                {
                    // adding the point will give a smaller diameter
                    // than the best one we've seen so far - note
                    // which point it was in case it turns out to be the
                    // best overall.
                    minDiameter = testDiameter;
                    minDiameterIdx = j;
                }
            }
        }

        // Now that we have examined all available points, see
        // if any of them resulted in a cluster smaller than the treshold.
        // NOTE: to avoid taking square roots on the Euclidean distance
        // between points, our diameters are "distance squared," so we
        // must test against "threshold squared."
        if( minDiameter > (threshold*threshold) )
        {
            // After examining all available points, none of them
            // result in a cluster that is smaller than the threshold.
            // Don't add any (more) points.
            break;
        }

        // add the point to the cluster, and keep looking to add more
        isPointInCluster[minDiameterIdx] = 1;
        work[2*size] = points[2*minDiameterIdx];
        work[2*size+1] = points[2*minDiameterIdx + 1];
        size++;
    }

    return size;
}



int
QT2( float* points,
        unsigned int numPoints,
        float threshold,
        unsigned int* clusterMap,
        unsigned int numClusters,
        unsigned int* pointMap )
{
    unsigned int* clusterSizes = (unsigned int*)malloc(numPoints * sizeof(unsigned int));

    unsigned int maxClusterSize = 0;
    unsigned int maxClusterIdx = 0;

#if READY
#else
    fprintf( stderr, "Entered QT2, %d clusters known, %d points available\n", numClusters, numPoints );
#endif // READY

// #pragma omp parallel
    {
        int* local_isPointInCluster = (int*)malloc( numPoints * sizeof(int) );
        float* local_work = (float*)malloc( numPoints * 2 * sizeof(float) );
        int myid = omp_get_thread_num();

        // For each point, find the largest cluster we can when using that
        // point as a seed.
// #pragma omp for
        for( unsigned int i = 0; i < numPoints; i++ )
        {
            clusterSizes[i] = FindCluster(points, 
                                            i, 
                                            local_isPointInCluster, 
                                            local_work, 
                                            numPoints, 
                                            threshold);
        }
        free( local_isPointInCluster );
        free( local_work );
    }

#if READY
    add openacc pragma with max reduce on this reduce(max : maxClusterSize)
    then linear search for idx that gives that max value
#endif // READY
        for( unsigned int i = 0; i < numPoints; i++ )
        {
            if( clusterSizes[i] > maxClusterSize )
            {
                maxClusterSize = clusterSizes[i];
                maxClusterIdx = i;
            }
        }

#if READY
    fprintf( stderr, "Cluster seeded at %d is largest (size %d)\n", maxClusterIdx, maxClusterSize );
#endif // READY

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
#if READY
#else
    if( clusterSize != maxClusterSize )
    {
        fprintf(stderr, "clusterSize=%d != maxClusterSize=%d\n", clusterSize, maxClusterSize);
    }
#endif // READY
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

#if READY
    fprintf( stderr, "Left QT2 (level %d), %d clusters known\n", numClusters );
#endif // READY
    return numClusters;
}


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
    for( unsigned int i = 0; i < numPoints; i++ )
    {
        clusterMap[i] = 0;
        pointMap[i] = i;
    }


    int clusteringTimerHandle = Timer_Start();
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


