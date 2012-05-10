#!/bin/sh

CPPFLAGS="-I/opt/cuda-4.2/cuda/include/" \
CUDA_CPPFLAGS="-gencode=arch=compute_20,code=sm_20" \
sh ./configure

