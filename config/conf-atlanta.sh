#!/bin/sh

OCL_ROOT=/opt/AMDAPP

# do the actual configuration
sh ./configure \
    CPPFLAGS="-I$OCL_ROOT/include" \
    CXXFLAGS="-g -O2" \
    CFLAGS="-g -O2" \
    LDFLAGS="-g -O2 -L$OCL_ROOT/lib/x86_64" \
    --without-cuda \
    --disable-stability

