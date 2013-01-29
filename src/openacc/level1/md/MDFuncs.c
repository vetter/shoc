#include <stdio.h>
#include "CTimer.h"

#include "constants.h"

void ljSingle(const unsigned int nIters,
        const int nAtom,
        const int maxNeighbors,
        float3* force,
        const float4* position,
        const int*    neighborList,
        double* itersTime,
        double* totalTime)
{

    // Start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    #pragma acc data copyin(position[0:nAtom], \
        neighborList[0:(nAtom*maxNeighbors)])              \
        copy(force[0:nAtom])
    {

    // Now that we've copied data to device, time the iterations.
    int iterTimerHandle = Timer_Start();

        // Note: do *not* try to map the iterations loop to the accelerator
        for( unsigned int iter = 0; iter < nIters; iter++ )
        {
            #pragma acc kernels loop
            for (int i = 0; i < nAtom; i++)
            {
                float4 ipos = position[i];
                float3 f = {0.0f, 0.0f, 0.0f};
             
                for (int j = 0; j < maxNeighbors; j++)
                {
                    int jidx = neighborList[j*nAtom + i];
                    float4 jpos = position[jidx];
                    
                    // Calculate distance
                    float delx = ipos.x - jpos.x;
                    float dely = ipos.y - jpos.y;
                    float delz = ipos.z - jpos.z;
                    float r2inv = delx*delx + dely*dely + delz*delz;

                    // If distance is less than cutoff, calculate force
                    if (r2inv < cutsq) 
                    {
                        r2inv = 1.0f/r2inv;
                        float r6inv = r2inv * r2inv * r2inv;
                        float force = r2inv*r6inv*(lj1*r6inv - lj2);

                        f.x += delx * force;
                        f.y += dely * force;
                        f.z += delz * force;
                    }
                } // for neighbors         
                force[i] = f;
            } // for atoms
        } // for iters
        *itersTime = Timer_Stop(iterTimerHandle, "");
    } // acc data
    *totalTime = Timer_Stop(wholeTimerHandle, "");
}

void ljDouble(const unsigned int nIters,
        const int nAtom,
        const int maxNeighbors,
        double3* force,
        const double4* position,
        const int*    neighborList,
        double* itersTime,
        double* totalTime)
{

    // Start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    #pragma acc data copyin(position[0:nAtom], \
        neighborList[0:(nAtom*maxNeighbors)])              \
        copy(force[0:nAtom])
    {

    // Now that we've copied data to device, time the iterations.
    int iterTimerHandle = Timer_Start();

        // Note: do *not* try to map the iterations loop to the accelerator
        for( unsigned int iter = 0; iter < nIters; iter++ )
        {
            #pragma acc kernels loop
            for (int i = 0; i < nAtom; i++)
            {
                double4 ipos = position[i];
                double3 f = {0.0f, 0.0f, 0.0f};
             
                for (int j = 0; j < maxNeighbors; j++)
                {
                    int jidx = neighborList[j*nAtom + i];
                    double4 jpos = position[jidx];
                    
                    // Calculate distance
                    double delx = ipos.x - jpos.x;
                    double dely = ipos.y - jpos.y;
                    double delz = ipos.z - jpos.z;
                    double r2inv = delx*delx + dely*dely + delz*delz;

                    // If distance is less than cutoff, calculate force
                    if (r2inv < cutsq) 
                    {
                        r2inv = 1.0f/r2inv;
                        double r6inv = r2inv * r2inv * r2inv;
                        double force = r2inv*r6inv*(lj1*r6inv - lj2);

                        f.x += delx * force;
                        f.y += dely * force;
                        f.z += delz * force;
                    }
                } // for neighbors         
                force[i] = f;
            } // for atoms
        } // for iters
        *itersTime = Timer_Stop(iterTimerHandle, "");
    } // acc data
    *totalTime = Timer_Stop(wholeTimerHandle, "");
}


