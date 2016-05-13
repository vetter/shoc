#include <sstream>
#include "InvalidArgValue.h"

std::string
InvalidArgValue::GenerateErrorMessage( const std::string& _msg )
{
    std::ostringstream msgstr;
    msgstr << "invalid argument value: ";
    if( _msg.length() > 0 )
    {
        msgstr << _msg;
    }
    else
    {
        msgstr << "no further details available";
    }
    return msgstr.str();
}


InvalidArgValue::InvalidArgValue( const std::string& _msg )
  : std::runtime_error( GenerateErrorMessage(_msg) )
{
    // nothing else to do
}

