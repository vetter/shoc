#ifndef PMSMEMMGMT_H
#define PMSMEMMGMT_H

#include <stdlib.h>

// Programming Model-Specific Memory Management
// Some programming models for heterogeneous systems provide
// memory management functions for allocating memory on the host
// and on the device.  These functions provide an abstract interface
// to that programming-model-specific interface.

template<class T>
T*
pmsAllocHostBuffer( size_t nItems )
{
    return new T[nItems];
}


template<class T>
void
pmsFreeHostBuffer( T* buf )
{
    delete[] buf;
}

#endif // PMSMEMMGMT_H
