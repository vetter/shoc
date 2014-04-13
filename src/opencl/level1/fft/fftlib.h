#ifndef FFTLIB_H
#define FFTLIB_H

#include "OptionParser.h"

extern cl_device_id fftDev;
extern cl_context fftCtx;
extern cl_command_queue fftQueue;
extern Event fftEvent, ifftEvent, chkEvent;

struct cplxflt {
    float x;
    float y;
};

struct cplxdbl {
    double x;
    double y;
};

void init(OptionParser& op, bool dp);
void deinit();
void forward(void* work, int n_ffts);
void inverse(void* work, int n_ffts);
int check(void* work, void* check, int half_n_ffts, int half_n_cmplx);
void allocDeviceBuffer(void** bufferp, unsigned long bytes);
void freeDeviceBuffer(void* buffer);
void allocHostBuffer(void** bufp, unsigned long bytes);
void freeHostBuffer(void* buf);
void copyToDevice(void* to_device, void* from_host, unsigned long bytes);
void copyFromDevice(void* to_host, void* from_device, unsigned long bytes);

#endif

