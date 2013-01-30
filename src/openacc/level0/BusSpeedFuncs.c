#include <stdio.h>
#include "CTimer.h"

void downloadFunc(const long long numBytes, float* data,
        double* totalTime) 
{
    const long long numFloats = numBytes / sizeof(float);
    #pragma acc data create(data[0:numFloats])
    {
        int updateTimerHandle = Timer_Start();
        #pragma acc update device(data[0:numFloats])
        *totalTime = Timer_Stop( updateTimerHandle, "" );
    } // end acc data create
}

void readbackFunc(const long long numBytes, float* data,
        double* totalTime) 
{
    const long long numFloats = numBytes / sizeof(float);
    #pragma acc data create(data[0:numFloats])
    {
        #pragma acc update device(data[0:numFloats])
        // Add a minimal kernel such that the update won't be
        // optimized away.
        #pragma acc kernels
        for (int i = 0; i < 1; i++) 
        {
            data[i] += 1.0f;
        }
        int updateTimerHandle = Timer_Start();
        #pragma acc update host(data[0:numFloats])
        *totalTime = Timer_Stop( updateTimerHandle, "" );
    } // end acc data create
}

