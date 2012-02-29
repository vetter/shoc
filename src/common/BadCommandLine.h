#ifndef BAD_COMMAND_LINE_H
#define BAD_COMMAND_LINE_H

#include <stdexcept>

// ****************************************************************************
// Class:  BadCommandLine
//
// Purpose:
//   Exception for command line parse errors
//
// Programmer:  Phil Roth
// Creation:    October 28, 2009
//
// ****************************************************************************
struct BadCommandLine : public std::exception
{
    // NOTE: current OptionParser implementation prints problems rather
    // than leaving it for us to determine how to print, so we have nothing
    // else to do.
    virtual char const* what( void ) const throw()  { return "invalid command line"; }
};

#endif // BAD_COMMAND_LINE_H
