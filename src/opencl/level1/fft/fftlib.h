#ifndef FFTLIB_H
#define FFTLIB_H

#include "OptionParser.h"

struct cplxflt {
    float x;
    float y;
};

struct cplxdbl {
    double x;
    double y;
};

void init(OptionParser& op,
     bool _do_dp,
     cl_device_id fftDev,
     cl_context fftCtx,
     cl_command_queue fftQueue,
     cl_program& fftProg,
     cl_kernel& fftKrnl,
     cl_kernel& ifftKrnl,
     cl_kernel& chkKrnl);

void deinit(cl_command_queue fftQueue,
            cl_program& fftProg,
            cl_kernel& fftKrnl,
            cl_kernel& ifftKrnl,
            cl_kernel& chkKrnl);

// Replaces forward and inverse, call with the
// appropriate kernel
void transform(void* workp,
              const int n_ffts,
              Event& fftEvent,
              cl_kernel& fftKrnl,
              cl_command_queue& fftQueue);

int check(const void* work,
          const void* check,
          const int half_n_ffts,
          const int half_n_cmplx,
          cl_kernel& chkKrnl,
          cl_command_queue& fftQueue);

void allocDeviceBuffer(void** bufferp,
                       const unsigned long bytes,
                       cl_context fftCtx,
                       cl_command_queue fftQueue);

void freeDeviceBuffer(void* buffer,
                      cl_context fftCtx,
                      cl_command_queue fftQueue);

void allocHostBuffer(void** bufp,
                     const unsigned long bytes,
                     cl_context fftCtx,
                     cl_command_queue fftQueue);

void freeHostBuffer(void* buf,
                    cl_context fftCtx,
                    cl_command_queue fftQueue);

void copyToDevice(void* to_device, void* from_host,
    const unsigned long bytes, cl_command_queue fftQueue);

void copyFromDevice(void* to_host, void* from_device,
    const unsigned long bytes, cl_command_queue fftQueue);

#endif // FFTLIB_H
