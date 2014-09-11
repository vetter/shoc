#include "Event.h"
#include "support.h"

using namespace std;

const int Event::ALL_EVENTS = -1;

Event::Event(const string &n, const int _count) : name(n),
                                count(_count),
                                queuedTime(0),
                                submitTime(0),
                                startTime(0),
                                endTime(0)
{
   event = new cl_event[count];
   queuedTime = new cl_ulong[count];
   submitTime = new cl_ulong[count];
   startTime = new cl_ulong[count];
   endTime = new cl_ulong[count];
   for (int i=0 ; i<count ; ++i) {
      event[i] = NULL;
      queuedTime[i] = 0;
      submitTime[i] = 0;
      startTime[i] = 0;
      endTime[i] = 0;
   }
}

Event::~Event()
{
   for (int i=0 ; i<count ; ++i)
      if (event[i])
         clReleaseEvent(event[i]);
   delete[] event;
   delete[] queuedTime;
   delete[] submitTime;
   delete[] startTime;
   delete[] endTime;
}


cl_event &Event::CLEvent(const int idx)
{
    return event[idx];
}

const cl_event &Event::CLEvent(const int idx) const
{
    return event[idx];
}

cl_event* Event::CLEvents()
{
    return event;
}

const cl_event* Event::CLEvents() const
{
    return event;
}


void Event::FillTimingInfo(const int idx)
{
    int sidx, eidx;
    if (idx == ALL_EVENTS) {
       sidx = 0; eidx = count-1;
    } else
       sidx = eidx = idx;
    for (int i=sidx ; i<=eidx ; ++i) {
       cl_int err;
       err = clGetEventProfilingInfo(event[i], CL_PROFILING_COMMAND_QUEUED,
                                     sizeof(cl_ulong), &queuedTime[i], NULL);
       CL_CHECK_ERROR(err);
       err = clGetEventProfilingInfo(event[i], CL_PROFILING_COMMAND_SUBMIT,
                                     sizeof(cl_ulong), &submitTime[i], NULL);
       CL_CHECK_ERROR(err);
       err = clGetEventProfilingInfo(event[i], CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &startTime[i], NULL);
       CL_CHECK_ERROR(err);
       err = clGetEventProfilingInfo(event[i], CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &endTime[i], NULL);
       CL_CHECK_ERROR(err);
    }
}

cl_ulong Event::QueuedTime(const int idx) const
{
    return queuedTime[idx];
}

cl_ulong Event::SubmitTime(const int idx) const
{
    return submitTime[idx];
}

cl_ulong Event::StartTime(const int idx) const
{
    return startTime[idx];
}

cl_ulong Event::EndTime(const int idx) const
{
    return endTime[idx];
}

cl_ulong Event::QueueSubmitDelay(const int idx) const
{
    return submitTime[idx] - queuedTime[idx];
}

cl_ulong Event::SubmitStartDelay(const int idx) const
{
    return startTime[idx] - submitTime[idx];
}

cl_ulong Event::SubmitEndRuntime(const int idx) const
{
    return endTime[idx] - submitTime[idx];
}

cl_ulong Event::StartEndRuntime(const int idx) const
{
    return endTime[idx] - startTime[idx];
}

cl_ulong Event::FullOverheadRuntime(const int idx) const
{
    return endTime[idx] - queuedTime[idx];
}

void Event::Print(ostream &out, const int idx) const
{
    int sidx, eidx;
    if (idx == ALL_EVENTS) {
       sidx = 0; eidx = count-1;
    } else
       sidx = eidx = idx;
    for (int i=sidx ; i<=eidx ; ++i) {
       out << "--> Event id=" << event[i] << ": " << name << " <--" << endl;
       out << "  raw queuedTime ns = " << queuedTime[i] << endl;
       out << "  raw submitTime ns = " << submitTime[i] << endl;
       out << "  raw startTime ns  = " << startTime[i] << endl;
       out << "  raw endTime ns    = " << endTime[i] << endl;

       out << "  queued-submit delay  = " << QueueSubmitDelay(i)/1.e6    << " ms\n";
       out << "  submit-start delay   = " << SubmitStartDelay(i)/1.e6    << " ms\n";
       out << "  start-end runtime    = " << StartEndRuntime(i)/1.e6     << " ms\n";
       out << "  queue-end total time = " << FullOverheadRuntime(i)/1.e6 << " ms\n";

       out << endl;
    }
}
