MonteCarlo is an offload version of MonteCarlo European options
It prices the European options using statistical computing method.
The program builds on Intel(r) Xeon(r) processor and offloads to 
Intel(r) Xeon Phi(tm) Coprocessors.

Its pathlength is 256*1024=262144.  The problem size is 64*1024*60
approximately each thread runs 64K data inputs.
The executible files are created using Intel Parallel Composer XE 2013 
sp1. It requires libiomp5.so from the same tools release.

To run the program in performance mode,set the environment variable

OFFLOAD_INIT=on_start

[...]$ ./MonteCarlo
Monte Carlo European Option Pricing in Single Precision
Pricing 3932160 options with path length of 262144.
Completed in   9.1321 seconds.
Computation rate - 430588.260 options per second.

When the program is passed the optional parameter 1 (--validate 1), it validates the result. 
[...]$ ./MonteCarlo 1
Monte Carlo European Option Pricing in Single Precision
...generating the input data.
Pricing 3932160 options with path length of 262144.
Completed in   9.1503 seconds.
Computation rate - 429731.067 options per second.
L1 norm: 1.061713E-03
Average reserve: 1.456985
...freeing CPU memory.
PASSED

The program is capable of reaching 430K opt/sec on KNC SE10p with 61 cores;
to build the binary on less than 61 core, modify OPT_N in MonteCarlo.h as
a guideline set OPT_N = number_of_working_cores*64*1024
for example on KNC SE10p with 61 cores, number_of_working_cores=60 so
const int OPT_N = 2*512*64*60; in MonteCarlo.h
