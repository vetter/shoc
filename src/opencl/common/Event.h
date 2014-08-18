#ifndef EVENT_H
#define EVENT_H

#include <string>
#include "support.h"


// ****************************************************************************
// Class:  Event
//
// Purpose:
//   Timing information for execution of a collection of OpenCL kernels.
//
// Programmer:  Jeremy Meredith
// Creation:    June 12, 2009
//
// ****************************************************************************
class Event
{
  public:
    /// Value to use for specifying to query all OpenCL kernel
    // events asssociated with this Event.
    static const int ALL_EVENTS;

  private:
    // Name of this collection of OpenCL kernel events.
    std::string name;

    // OpenCL-provided data regarding this collection of OpenCL kernels.
    cl_event    *event;

    // Time that each OpenCL kernel in this collection were queued.
    cl_ulong    *queuedTime;

    // Time that each OpenCL kernel in this collection were submitted to the device.
    cl_ulong    *submitTime;

    // Time that each OpenCL kernel in this collection started running on the device.
    cl_ulong    *startTime;

    // Time that each OpenCL kernel in this collection finished running on the device.
    cl_ulong    *endTime;

    // Number of OpenCL kerenls represented in this Event's collection.
    cl_int      count;

  public:
    Event(const std::string &name, const int _count = 1);
    ~Event();


    // Retrieve OpenCL data associated with this Event.
    cl_event        &CLEvent(const int idx = 0);
    const cl_event  &CLEvent(const int idx = 0) const;

    // Retrieve OpenCL data for each OpenCL kernel associated with this Event.
    cl_event*       CLEvents();
    const cl_event* CLEvents() const;

    // Obtain timing information from OpenCL runtime for each kernel
    // associated with this Event.
    void            FillTimingInfo(const int idx = 0);

    // Retrieve the timestamp that an associated OpenCL kernel was enqueued.
    cl_ulong        QueuedTime(const int idx = 0) const;

    // Retrieve the timestamp that an associated OpenCL kernel was submitted to the device.
    cl_ulong        SubmitTime(const int idx = 0) const;

    // Retrieve the timestamp that an associated OpenCL kernel started running on the device.
    cl_ulong        StartTime(const int idx = 0) const;

    // Retrieve the timestamp that an associated OpenCL kernel finished running on the device.
    cl_ulong        EndTime(const int idx = 0) const;

    // Retrieve the time that an OpenCL kernel spent enqueued.
    cl_ulong        QueueSubmitDelay(const int idx = 0) const;

    // Retrieve the time that an OpenCL kernel spent on the device before starting.
    cl_ulong        SubmitStartDelay(const int idx = 0) const;

    // Retrieve the time between when a kernel was submitted to the device and
    // when it finishes executing
    cl_ulong        SubmitEndRuntime(const int idx = 0) const;

    // Retrieve the time that an OpenCL kernel spent executing on the device.
    cl_ulong        StartEndRuntime(const int idx = 0) const;

    // Retrieve the total time required to run an OpenCL kernel.
    cl_ulong        FullOverheadRuntime(const int idx = 0) const;

    // Dump timing information for kernels associated with this Event to the given stream.
    void            Print(std::ostream&, const int idx = 0) const;
};


#endif
