#ifndef PMSMEMMGMT_H
#define PMSMEMMGMT_H

#include <stdlib.h>

// Programming Model-Specific Memory Management
// Some programming models for heterogeneous systems provide 
// memory management functions for allocating memory on the host
// and on the device.  These functions provide an abstract interface
// to that programming-model-specific interface.

#ifdef USE_MM_MALLOC

#define ALIGN   4096

template<class T>
T*
pmsAllocHostBuffer( size_t nItems )
{
    return (T*)_mm_malloc(nItems * sizeof(T), ALIGN);
}

template<class T>
void
pmsFreeHostBuffer( T* buf )
{
    _mm_free(buf);
}

#else
    // use regular old C++ array new and delete operations

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
#endif // USE_MM_MALLOC

#endif // PMSMEMMGMT_H
