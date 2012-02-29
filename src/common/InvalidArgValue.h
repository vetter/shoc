#ifndef INVALIDARGVALUE_H
#define INVALIDARGVALUE_H

#include <stdexcept>

// Exception for command line argument value errors
class InvalidArgValue : public std::exception
{
private:
    static const char* defMsg;
    char* msg;

public:
    InvalidArgValue( const char* _msg );
    virtual ~InvalidArgValue( void ) throw();
    virtual char const* what( void ) const throw();

    const char* GetMessage( void ) const    { return (msg != NULL) ? msg : defMsg; }
};

#endif // INVALIDARGVALUE_H
