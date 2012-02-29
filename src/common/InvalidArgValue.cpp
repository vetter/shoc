#include <string.h>
#include "InvalidArgValue.h"

const char* InvalidArgValue::defMsg = "invalid argument value: no further details available";


InvalidArgValue::InvalidArgValue( const char* _msg )
  : msg( NULL )
{
    if( _msg != NULL )
    {
        try
        {
            msg = new char[strlen(_msg)];
            strcpy( msg, _msg );
        }
        catch(...)
        {
            // nothing else to do - just leave msg as NULL
        }
    }
}

InvalidArgValue::~InvalidArgValue( void ) throw ()
{
    delete[] msg;
    msg = NULL;
}


char const*
InvalidArgValue::what( void ) const throw()
{
    return "invalid argument value";
}

