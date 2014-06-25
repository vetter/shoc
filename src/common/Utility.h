#ifndef UTILITY_H
#define UTILITY_H

#include <sstream>
#include <math.h>

// ****************************************************************************
// File:  Utility.h
//
// Purpose:
//   Various generic utility routines having to do with string and number
//   manipulation.
//
// Programmer:  Jeremy Meredith
// Creation:    September 18, 2009
// Modified:    Jan 2010, rothpc
//    Jeremy Meredith, Tue Oct  9 17:25:25 EDT 2012
//    Round is c99, not Windows-friendly.  Assuming we are using
//    positive values, replaced it with an equivalent of int(x+.5).
//
// ****************************************************************************

inline std::string HumanReadable(long long value, long long *rounding=0)
{
    std::ostringstream vstr;
    long long pVal;
    if (value>10ll*1024*1024*1024)
    {
        pVal = (long long)(0.5 + value/(1024.0*1024*1024));
        if (rounding)
            *rounding = pVal*1024*1024*1024 - value;
        vstr << pVal << 'G';
    }
    else if (value>10ll*1024*1024)
    {
        pVal = (long long)(0.5 + value/(1024.0*1024));
        if (rounding)
            *rounding = pVal*1024*1024 - value;
        vstr << pVal << 'M';
    }
    else if (value>10ll*1024)
    {
        pVal = (long long)(0.5 + value/(1024.0));
        if (rounding)
            *rounding = pVal*1024 - value;
        vstr << pVal << 'k';
    }
    else
    {
        if (rounding)
            *rounding = 0;
        vstr << value;
    }
    return vstr.str();
}

inline vector<string> SplitValues(const std::string &buff, char delim)
{
    vector<std::string> output;
    std::string tmp="";
    for (size_t i=0; i<buff.length(); i++)
    {
       if (buff[i] == delim)
       {
          if (!tmp.empty())
             output.push_back(tmp);
          tmp = "";
       }
       else
       {
          tmp += buff[i];
       }
    }
    if (!tmp.empty())
       output.push_back(tmp);

    return output;
}

#ifdef _WIN32

// On Windows, srand48 and drand48 don't exist.
// Create convenience routines that use srand/rand
// and let developers continue to use the -48 versions.

inline void srand48(unsigned int seed)
{
    srand(seed);
}

inline double drand48()
{
    return double(rand()) / RAND_MAX;
}

#endif // _WIN32

#endif
