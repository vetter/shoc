#include <cassert>
#include "qtclib.h"

void
allocHostBuffer( void** buf, unsigned long nBytes )
{
    char* cbuf = new char[nBytes];

    assert( buf != 0 );
    *buf = static_cast<void*>( cbuf );
}

void
freeHostBuffer( void* buf )
{
    char* cbuf = static_cast<char*>( buf );
    delete[] cbuf;
}


