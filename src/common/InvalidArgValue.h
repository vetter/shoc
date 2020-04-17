#ifndef INVALIDARGVALUE_H
#define INVALIDARGVALUE_H

#include <stdexcept>
#include <string>

// Exception for command line argument value errors
class InvalidArgValue : public std::runtime_error
{
private:
    static std::string GenerateErrorMessage( const std::string& _msg );

public:
    InvalidArgValue( const std::string& _msg );
};

#endif // INVALIDARGVALUE_H
