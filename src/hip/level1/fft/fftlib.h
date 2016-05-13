#ifndef FFTLIB_H
#define FFTLIB_H

#include "OptionParser.h"

extern int fftDevice;

void init(OptionParser& op, const bool do_dp, const int n_ffts);
void forward(void* work, const int n_ffts);
void inverse(void* work, const int n_ffts);
int check(void* work, void* check, const int half_n_ffts, 
    const int half_n_cmplx);
void allocHostBuffer(void** bufferp, const unsigned long bytes);
void allocDeviceBuffer(void** bufferp, const unsigned long bytes);
void freeHostBuffer(void* buffer);
void freeDeviceBuffer(void* buffer);
void copyToDevice(void* to_device, const void* from_host, 
    const unsigned long bytes);
void copyFromDevice(void* to_host, const void* from_device, 
    const unsigned long bytes);

#endif // FFTLIB_H
