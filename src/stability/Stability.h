#ifndef STABILITY_H

void init(OptionParser& op);
void forward(void* work, const int n_tasks);
void inverse(void* work, const int n_tasks);
int check(void* work, void* check, const int half_n_tasks, const int half_n_elts);
unsigned long findAvailBytes(void);
void allocDeviceBuffer(void** bufferp, const unsigned long bytes);
void freeDeviceBuffer(void* buffer);
void copyToDevice(void* to_device, const void* from_host, const unsigned long bytes);
void copyFromDevice(void* to_host, const void* from_device, const unsigned long bytes);

#endif

