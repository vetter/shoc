#!/bin/sh

sh ./configure \
CPPFLAGS="-I/sw/kfs/cuda/4.2/linux_binary/include" \
CUDA_CPPFLAGS="-gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20"

