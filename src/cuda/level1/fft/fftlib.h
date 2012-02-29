#ifndef FFTLIB_H
#define FFTLIB_H

#include "OptionParser.h"

extern int fftDevice;

void init(OptionParser& op, bool do_dp);
void forward(void* work, int n_ffts);
void inverse(void* work, int n_ffts);
int check(void* work, void* check, int half_n_ffts, int half_n_cmplx);
void allocHostBuffer(void** bufferp, unsigned long bytes);
void allocDeviceBuffer(void** bufferp, unsigned long bytes);
void freeHostBuffer(void* buffer);
void freeDeviceBuffer(void* buffer);
void copyToDevice(void* to_device, void* from_host, unsigned long bytes);
void copyFromDevice(void* to_host, void* from_device, unsigned long bytes);

#endif // FFTLIB_H
