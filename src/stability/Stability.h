#ifndef STABILITY_H

void init(OptionParser& op);
void forward(void* work, int n_tasks);
void inverse(void* work, int n_tasks);
int check(void* work, void* check, int half_n_tasks, int half_n_elts);
unsigned long findAvailBytes(void);
void allocDeviceBuffer(void** bufferp, unsigned long bytes);
void freeDeviceBuffer(void* buffer);
void copyToDevice(void* to_device, void* from_host, unsigned long bytes);
void copyFromDevice(void* to_host, void* from_device, unsigned long bytes);

#endif

