#!/bin/sh

OCL_ROOT=/opt/AMDAPP

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


# do the actual configuration
sh ./configure \
    CPPFLAGS="-I$OCL_ROOT/include" \
    CXXFLAGS="$OPTFLAGS" \
    CFLAGS="$OPTFLAGS" \
    LDFLAGS="$OPTFLAGS -L$OCL_ROOT/lib/x86_64" \
    --without-cuda \
    --disable-stability

