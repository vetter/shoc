#!/bin/sh

OCL_ROOT=/opt/AMDAPP

# A "good" set of optimization flags is compiler dependent.
# These might be reasonable flags to start from.
#
# GNU
OPTFLAGS="-g -O2"

# Intel
#OPTFLAGS="-g -fast"

# PGI
#OPTFLAGS="-g -fastsse"


# do the actual configuration
sh ./configure \
    CPPFLAGS="-I$OCL_ROOT/include" \
    CXXFLAGS="$OPTFLAGS" \
    CFLAGS="$OPTFLAGS" \
    LDFLAGS="$OPTFLAGS -L$OCL_ROOT/lib/x86_64" \
    --without-cuda \
    --disable-stability

