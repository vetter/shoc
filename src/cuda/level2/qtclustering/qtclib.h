#ifndef QTLIB_H
#define QTLIB_H

#include "OptionParser.h"

extern int qtcDevice;

void init(OptionParser& op);
void reduce_card(void *card, int pointCount);
void allocHostBuffer(void** bufferp, unsigned long bytes);
void allocDeviceBuffer(void** bufferp, unsigned long bytes);
void freeHostBuffer(void* buffer);
void freeDeviceBuffer(void* buffer);
void copyToDevice(void* to_device, void* from_host, unsigned long bytes);
void copyFromDevice(void* to_host, void* from_device, unsigned long bytes);

#endif // QTLIB_H
