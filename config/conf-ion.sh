#!/bin/sh

# "ion" is an Ubuntu 12.04 system with gcc 4.6.x

PATH="/usr/local/cuda42/cuda/bin:$PATH"
CPPFLAGS="-I/usr/local/cuda42/cuda/include/" \
CUDA_CPPFLAGS="-gencode=arch=compute_20,code=sm_20 -DUSE_CLOCK_GETTIME" \
CUDA_LDFLAGS="-Xlinker -rpath=/usr/local/cuda42/cuda/lib64" \
sh ./configure

