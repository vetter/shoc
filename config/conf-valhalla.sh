#!/bin/sh
# note: "valhalla" is an Ubuntu 12.04 system with gcc 4.6.x

which nvcc
if (test $? -ne 0); then
   echo "Error: no nvcc found.  Please set your path:"
   echo "export PATH=\"/usr/local/cuda-6.0/bin:\$PATH\""
   echo "export LD_LIBRARY_PATH=\"/usr/local/cuda-6.0/lib64:\$PATH\""
   exit 1
fi

./configure \
CPPFLAGS="-I/usr/local/cuda-6.0/include/" \
CUDA_CPPFLAGS="-DUSE_CLOCK_GETTIME -gencode=arch=compute_50,code=sm_50"

