#!/bin/sh


# A "good" set of optimization flags is compiler dependent.
# These might be reasonable flags to start from.
#
# GNU
OPTFLAGS="-g -O2"

# Intel
#OPTFLAGS="-g -xHOST -O3 -ip -no-prec-div"

# PGI
#OPTFLAGS="-g -fastsse"

CXX=icpc \
CC=icc \
CPPFLAGS="-I/opt/cuda/4.2/cuda/include/" \
CUDA_CPPFLAGS="-gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30" \
sh ./configure \
    CXXFLAGS="$OPTFLAGS" \
    CFLAGS="$OPTFLAGS" \
    LDFLAGS="$OPTFLAGS"

