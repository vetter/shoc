#ifndef QTCLIB_H
#define QTCLIB_H

// QTC support functions for OpenACC version.
// Note: other programming systems (e.g., CUDA) have different collections
// of functions defined in this interface.

void allocHostBuffer( void** bufferp, unsigned long nBytes );
void freeHostBuffer( void* buffer );

#endif // QTCLIB_H
