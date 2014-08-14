#!/bin/sh


# A "good" set of optimization flags is compiler dependent.
# These might be reasonable flags to start from.
#
# GNU
PATH="/opt/cuda/6.0/cuda/bin:$PATH"
OPTFLAGS="-g -O2"

# Intel
#OPTFLAGS="-g -xHOST -O3 -ip -no-prec-div"
#export CXX=icpc
#export CC=icc

# PGI
#OPTFLAGS="-g -fastsse"
#export CXX=pgcpp
#export CC=pgcc

CPPFLAGS="-I/opt/cuda/6.0/cuda/include/" \
CUDA_CPPFLAGS="-gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30" \
sh ./configure \
    CXXFLAGS="$OPTFLAGS" \
    CFLAGS="$OPTFLAGS" \
    LDFLAGS="$OPTFLAGS"

