#!/bin/sh

# Configure to build OpenCL and CUDA tests.

# By default, we build 64-bit executables on OS X because CUDA 4.0 and later
# provide 64-bit support for all libraries.  
#
# However, if you are using an earlier version of CUDA than 4.0, or
# are on a Mac without a x86_64 processor, you can change the -m64 flag
# to be -m32 in the configure script below to build 32-bit executables 
# (assuming you are building with gcc - use whatever flags are necessary 
# for your compiler).  For example:
#sh ./configure \
#    CXXFLAGS="-m32" \
#    CFLAGS="-m32" \
#    NVCXXFLAGS="-m32" \
#    --with-opencl --with-cuda 


 sh ./configure \
    CXXFLAGS="-m64" \
    CFLAGS="-m64" \
    NVCXXFLAGS="-m64" \
    --with-opencl --with-cuda

