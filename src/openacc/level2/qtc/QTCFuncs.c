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
DistanceSquared( Point* a, Point* b )
{
    float dx = (a->x - b->x);
    float dy = (a->y - b->y);
    return (dx*dx + dy*dy);
}


// Find the diameter of the given cluster if the testPoint were added to it.
// Note: to avoid having to compute square roots, our distances are 
// "distance squared," so the diameter value reported by this function is 
// actually a distance squared.
float
Diameter(Point* cluster, unsigned int size, Point* testPoint)
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
FindCluster(Point* points,
                unsigned int seedIdx, 
                int* isPointInCluster, 
                Point* work, 
                unsigned int numPoints, 
                float threshold)
{
    // at the start, only the seed point is in the cluster
    unsigned int size = 1;
    memset(isPointInCluster, 0, numPoints*sizeof(int));
    isPointInCluster[seedIdx] = 1;

    work[0] = points[seedIdx];

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
                float testDiameter = Diameter(work, size, &(points[j]) );
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
        work[size] = points[minDiameterIdx];
        size++;
    }

    return size;
}



void
DoFloatQTC( float* pointsAsFloats,
            unsigned int numPoints,
            float threshold,
            double* clusteringTime,
            double* totalTime )
{
    unsigned int numPointsTotal = numPoints;
    unsigned int numClusters = 0;
    unsigned int overallMaxClusterSize = 0;
    Point* points = (Point*)pointsAsFloats;

    int totalTimerHandle = Timer_Start();

    unsigned int* clusterMap = (unsigned int*)malloc(numPoints * sizeof(unsigned int));
    unsigned int* pointMap = (unsigned int*)malloc(numPoints * sizeof(unsigned int));
    for( unsigned int i = 0; i < numPoints; i++ )
    {
        clusterMap[i] = 0;
        pointMap[i] = i;
    }


    int clusteringTimerHandle = Timer_Start();
    while( numPoints > 0 )
    {
        unsigned int* clusterSizes = (unsigned int*)malloc(numPoints * sizeof(unsigned int));

        unsigned int maxClusterSize = 0;
        unsigned int maxClusterIdx = 0;


    // #pragma omp parallel
        {
            int* local_isPointInCluster = (int*)malloc( numPoints * sizeof(int) );
            Point* local_work = (Point*)malloc( numPoints * sizeof(Point) );
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
    #else
        fprintf( stderr, "Cluster seeded at %d is largest (size %d)\n", maxClusterIdx, maxClusterSize );
    #endif // READY

        int* isPointInCluster = (int*)malloc( numPoints * sizeof(int) );
        Point* work = (Point*)malloc( numPoints * sizeof(Point));
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

        numClusters++;
        if( clusterSize > overallMaxClusterSize )
        {
            overallMaxClusterSize = clusterSize;
        }


        // Remove the cluster we just found from the set of points.
        unsigned int numPointsRemaining = numPoints - clusterSize;

        if( numPointsRemaining > 0 )
        {
            // build a new points array for the remaining points
            Point* newPoints = (Point*)malloc( numPointsRemaining * sizeof(Point) );
            unsigned int* newPointMap = (unsigned int*)malloc( numPointsRemaining * sizeof(unsigned int));

            unsigned int pointsHandled = 0;
            for( unsigned int i = 0; i < numPoints; i++ )
            {
                if( !isPointInCluster[i] )
                {
                    newPointMap[pointsHandled] = pointMap[i];
                    newPoints[pointsHandled] = points[i];
                    pointsHandled++;
                }
                else
                {
                    // The point is in the newly-found cluster.
                    // Indicate its membership in the cluster map.
                    clusterMap[pointMap[i]] = numClusters;
                }
            }
            assert( pointsHandled == numPointsRemaining );

            free( points );
            free( pointMap );
            points = newPoints;
            pointMap = newPointMap;
        }

        numPoints = numPointsRemaining;
    }
    *clusteringTime = Timer_Stop( clusteringTimerHandle, "" );

    free( clusterMap );
    free( pointMap );

    *totalTime = Timer_Stop( totalTimerHandle, "" );

    fprintf(stdout, "%d clusters, max size %d\n",
        numClusters, 
        overallMaxClusterSize );
}



