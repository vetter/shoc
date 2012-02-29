#!/bin/sh

sh ./configure \
  CPPFLAGS="-I/opt/cuda-4.0/cuda/include" \
  LDFLAGS="-L/opt/cuda-4.0/cuda/lib64"    


# other useful options
#    --disable-stability

