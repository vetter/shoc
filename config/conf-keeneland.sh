#!/bin/sh

# A "good" set of optimization flags is compiler dependent.
# These might be reasonable flags to start from.
#
# GNU
OPTFLAGS="-g -O2"

# Intel
#OPTFLAGS="-g -xHOST -O3 -ip -no-prec-div"
#export CXX=icpc
#export CC=icc

# PGI
#OPTFLAGS="-g -fastsse"
#export CXX=pgcpp
#export CC=pgcc


sh ./configure \
    CPPFLAGS="-I/sw/kfs/cuda/4.2/linux_binary/include" \
    CUDA_CPPFLAGS="-gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20" \
    CXXFLAGS="$OPTFLAGS" \
    CFLAGS="$OPTFLAGS" \
    LDFLAGS="$OPTFLAGS"

