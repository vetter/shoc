#!/bin/sh

# Configure to build OpenCL and CUDA tests.

# By default, we build 32-bit executables on OS X because pre-4.0 versions
# of CUDA do not provide 64-bit support for all libraries.  
#
# However, if you are using CUDA 4.0 on a Mac with a x86_64 processor, you 
# can add -m64 to compile flags to build 64-bit executables (assuming you are
# building with gcc - use whatever flags are necessary for your compiler).  For
# example:
#
# sh ./configure \
#    CXXFLAGS="-m64" \
#    CFLAGS="-m64" \
#    NVCXXFLAGS="-m64" \
#    --with-opencl --with-cuda
#
sh ./configure \
    CXXFLAGS="-m32" \
    CFLAGS="-m32" \
    NVCXXFLAGS="-m32" \
    --with-opencl --with-cuda 

