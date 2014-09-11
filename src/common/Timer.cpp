#include "Timer.h"
#include <stdio.h>
#include <algorithm>

using std::cerr;
using std::endl;
using std::max;

// ----------------------------------------------------------------------------

Timer *Timer::instance = NULL;

// ----------------------------------------------------------------------------
static double
DiffTime(const struct TIMEINFO &startTime, const struct TIMEINFO &endTime)
{
#if defined(_WIN32)
    //
    // Figure out how many milliseconds between start and end times
    //
    int ms = (int) difftime(endTime.time, startTime.time);
    if (ms == 0)
    {
        ms = endTime.millitm - startTime.millitm;
    }
    else
    {
        ms =  ((ms - 1) * 1000);
        ms += (1000 - startTime.millitm) + endTime.millitm;
    }

    double seconds = (ms/1000.);
#elif defined(HAVE_CLOCK_GETTIME) && defined(HAVE_CLOCK_PROCESS_CPUTIME_ID)
    double seconds = double(endTime.tv_sec - startTime.tv_sec) +
                    double(endTime.tv_nsec - startTime.tv_nsec) / 1.0e9;

#elif defined(HAVE_GETTIMEOFDAY)

    double seconds = double(endTime.tv_sec - startTime.tv_sec) +
                     double(endTime.tv_usec - startTime.tv_usec) / 1000000.;

#else
#   error No supported timer available.
#endif
    return seconds;
}

static void
GetCurrentTimeInfo(struct TIMEINFO &timeInfo)
{
#if defined(_WIN32)
    _ftime(&timeInfo);
#elif defined(HAVE_CLOCK_GETTIME) && defined(HAVE_CLOCK_PROCESS_CPUTIME_ID)
    clock_gettime( CLOCK_REALTIME, &timeInfo );
#elif defined(HAVE_GETTIMEOFDAY)
    gettimeofday(&timeInfo, 0);
#else
#   error No supported timer available.
#endif
}



// ****************************************************************************
//  Constructor:  Timer::Timer
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
Timer::Timer()
{
    // Initialize some timer methods and reserve some space.
    startTimes.reserve(1000);
    timeLengths.reserve(1000);
    descriptions.reserve(1000);
    currentActiveTimers = 0;
}

// ****************************************************************************
//  Destructor:
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
Timer::~Timer()
{
    // nothing to do
}

// ****************************************************************************
//  Method:  Timer::Instance
//
//  Purpose:
//    Return the timer singleton.
//
//  Arguments:
//
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
Timer *Timer::Instance()
{
    if (!instance)
    {
        instance = new Timer;
    }
    return instance;
}

// ****************************************************************************
//  Method:  Timer::Start
//
//  Purpose:
//    Start a timer, and return a handle.
//
//  Arguments:
//    none
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
int Timer::Start()
{
    return Instance()->real_Start();
}

// ****************************************************************************
//  Method:  Timer::Stop
//
//  Purpose:
//    Stop a timer and add its length to our list.
//
//  Arguments:
//    handle       a timer handle returned by Timer::Start
//    desription   a description for the event timed
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
double Timer::Stop(int handle, const std::string &description)
{
    return Instance()->real_Stop(handle, description);
}

// ****************************************************************************
//  Method:  Timer::Insert
//
//  Purpose:
//    Add a user-generated (e.g. calculated) timing to the list
//
//  Arguments:
//    desription   a description for the event timed
//    value        the runtime to insert
//
//  Programmer:  Jeremy Meredith
//  Creation:    October 22, 2007
//
// ****************************************************************************
void Timer::Insert(const std::string &description, double value)
{
    Instance()->real_Insert(description, value);
}

// ****************************************************************************
//  Method:  Timer::Dump
//
//  Purpose:
//    Add timings to on ostream.
//
//  Arguments:
//    out        the stream to print to.
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
void Timer::Dump(std::ostream &out)
{
    return Instance()->real_Dump(out);
}

// ****************************************************************************
//  Method:  Timer::real_Start
//
//  Purpose:
//    the true start routine
//
//  Arguments:
//    none
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
int Timer::real_Start()
{
    int handle = startTimes.size();
    currentActiveTimers++;

    struct TIMEINFO t;
    GetCurrentTimeInfo(t);
    startTimes.push_back(t);

    return handle;
}

// ****************************************************************************
//  Method:  Timer::real_Stop
//
//  Purpose:
//    the true stop routine
//
//  Arguments:
//    handle       a timer handle returned by Timer::Start
//    desription   a description for the event timed
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
double Timer::real_Stop(int handle, const std::string &description)
{
    if ((unsigned int)handle > startTimes.size())
    {
        cerr << "Invalid timer handle '"<<handle<<"'\n";
        exit(1);
    }

    struct TIMEINFO t;
    GetCurrentTimeInfo(t);
    double length = DiffTime(startTimes[handle], t);
    timeLengths.push_back(length);

    char str[2048];
    sprintf(str, "%*s%s", currentActiveTimers*3, " ", description.c_str());
    descriptions.push_back(str);

    currentActiveTimers--;
    return length;
}

// ****************************************************************************
//  Method:  Timer::real_Insert
//
//  Purpose:
//    the true insert routine
//
//  Arguments:
//    desription   a description for the event timed
//    value        the run time to insert
//
//  Programmer:  Jeremy Meredith
//  Creation:    October 22, 2007
//
// ****************************************************************************
void Timer::real_Insert(const std::string &description, double value)
{
#if 0 // can disable inserting just to make sure it isn't broken
    cerr << description << " " << value << endl;
#else
    timeLengths.push_back(value);

    char str[2048];
    sprintf(str, "%*s[%s]",
            (currentActiveTimers+1)*3, " ", description.c_str());
    descriptions.push_back(str);
#endif
}

// ****************************************************************************
//  Method:  Timer::real_Dump
//
//  Purpose:
//    the true dump routine
//
//  Arguments:
//    out        the stream to print to.
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
void Timer::real_Dump(std::ostream &out)
{
    size_t maxlen = 0;
    for (unsigned int i=0; i<descriptions.size(); i++)
        maxlen = max(maxlen, descriptions[i].length());

    out << "\nTimings\n-------\n";
    for (unsigned int i=0; i<descriptions.size(); i++)
    {
        char desc[10000];
        sprintf(desc, "%-*s", (int)maxlen, descriptions[i].c_str());
        out << desc << " took " << timeLengths[i] << endl;
    }
}


