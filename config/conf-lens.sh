#!/bin/sh


CUDA_ROOT=/sw/analysis-x64/cuda/3.2/sl5.0_binary

# do the actual configuration
sh ./configure \
CPPFLAGS="-I$CUDA_ROOT/include" \
PATH"=$CUDA_ROOT/bin:$PATH" \
    --disable-stability


