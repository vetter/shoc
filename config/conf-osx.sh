#!/bin/sh

# Configure to build OpenCL and CUDA tests.

# By default, building on recent OS X systems will build 64-bit versions
# of all libraries and executables.
#
# However, if you are using an earlier version of CUDA than 4.0, or
# are on a Mac without a x86_64 processor, you can add the -m32 flag
# in the configure script below to build 32-bit executables 
# (assuming you are building with gcc - use whatever flags are necessary 
# for your compiler).  For example:
#sh ./configure \
#    CXXFLAGS="-m32" \
#    CFLAGS="-m32" \
#    NVCXXFLAGS="-m32" \
#    --with-opencl --with-cuda 

#
# On OS X 10.9 (Mavericks) , the Xcode toolchain defaults to using libc++
# as the C++ standard library.  CUDA 6.0's nvcc does not support libc++,
# so we have to specify to use libstdc++ instead.
#

#
# The gencode specification here is for a GPU with compute capability 3.0,
# such as a GeForce GT 750M in some recent MacBook Pro laptops.
# Modify it to suit your GPU's compute capability.
#


sh ./configure \
    CUDA_CPPFLAGS="-gencode=arch=compute_30,code=sm_30" \
    CXXFLAGS="-stdlib=libstdc++" \
    --with-opencl --with-cuda

# Example simple config for Mavericks (10.9.2) and CUDA 6.0rc, where
# driving with g++ can be problematic.
#sh ./configure \
#    CXX="nvcc" \
#    CPP="nvcc" \
#    --without-mpi \
#    --without-opencl --with-cuda

# Another issue on Mavericks (10.9.2) arises when compiling opencl with
# clang.  An alternative is to use gcc-4.8 (tested with the default config
# in homebrew) and the following:
#sh ./configure \
#  CXXFLAGS="-m64" \
#  CFLAGS="-m64" \
#  NVCXXFLAGS="-m64" \
#  CPP="g++-4.8" \
#  CXX="g++-4.8" \
#  --with-opencl --without-cuda --without-mpi


