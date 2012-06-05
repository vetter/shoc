#!/bin/sh

CPPFLAGS="-I/opt/cuda/4.2/cuda/include/" \
CUDA_CPPFLAGS="-gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30" \
sh ./configure

