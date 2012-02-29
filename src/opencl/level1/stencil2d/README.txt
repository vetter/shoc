
A relatively naive 9-point stencil operation over a 2D array.  The result
computed via OpenCL is compared against the result computed on the host CPU.

In the OpenCL implementation, a thread block copies data from the array in
device global memory to shared memory with a 1-element-wide halo, and each
thread in the thread block computes one data point of the array.

Double buffering in device global memory is used to avoid problems with mixing
'new' and 'old' array data, since there no synchronization across thread
blocks is available in the device code.

The number of iterations, the weights used in the stencil operation, and the
dimensions of the 2D array are all user configurable.

Better performance is likely possible.  Some potential optimizations include:
having a thread compute multiple data points; reducing the branching logic in
the code that loads shared memory.

