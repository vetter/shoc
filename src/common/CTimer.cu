#include "CTimer.h"
#include "Timer.h"

int
Timer_Start()
{
    return Timer::Start();
}

double
Timer_Stop(int h, const char *d)
{
    return Timer::Stop(h,d);
}

void
Timer_Insert(const char *d, double v)
{
    Timer::Insert(d,v);
}
