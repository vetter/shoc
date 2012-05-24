#!/bin/sh

# "ion" is an Ubuntu 12.04 system with gcc 4.6.x

PATH="/usr/local/cuda42/cuda/bin:$PATH"
CPPFLAGS="-I/usr/local/cuda42/cuda/include/" \
CUDA_CPPFLAGS="-DUSE_CLOCK_GETTIME -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30" \
sh ./configure

