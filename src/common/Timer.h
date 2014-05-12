#ifndef TIMER_H
#define TIMER_H

#include <vector>
#include <string>
#include <iostream>

#include <time.h>
#include <sys/timeb.h>
#ifndef _WIN32
#include <sys/time.h>
#include "config.h"
#endif


// decide which timer type we are supposed to use
#if defined(_WIN32)
#    define TIMEINFO _timeb
#elif defined(HAVE_CLOCK_GETTIME) && defined(HAVE_CLOCK_PROCESS_CPUTIME_ID)
#    define TIMEINFO timespec
#elif defined(HAVE_GETTIMEOFDAY)
#    define TIMEINFO timeval
#else
#    error No supported timer available.
#endif


// ****************************************************************************
//  Class:  Timer
//
//  Purpose:
//    Encapsulated a set of hierarchical timers.  Starting a timer
//    returns a handle to a timer.  Pass this handle, and a description,
//    into the timer Stop routine.  Timers can nest and output will
//    be displayed in a tree format.
//
//    Externally, Timer represents time in units of seconds.
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  6, 2004
//
// ****************************************************************************
class Timer
{
  public:
    static Timer *Instance();

    static int    Start();

    // Returns time since start of corresponding timer (determined by handle),
    // in seconds.
    static double Stop(int handle, const std::string &descr);
    static void   Insert(const std::string &descr, double value);

    static void   Dump(std::ostream&);

  private:

    int    real_Start();
    double real_Stop(int, const std::string &);
    void   real_Insert(const std::string &descr, double value);
    void   real_Dump(std::ostream&);

    Timer();
    ~Timer();

    static Timer *instance;

    std::vector<TIMEINFO>    startTimes;
    std::vector<double>      timeLengths;
    std::vector<std::string> descriptions;
    int                      currentActiveTimers;
};

#endif
