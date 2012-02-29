#!/bin/sh

#
# Configure SHOC for cross-compilation using ARM cross compilers.
# Gives an example of how to cross compile, should be adaptable to 
# other cross compilation targets.
#
# Assumes we are using CodeSourcery Lite ARM cross compilers.
# Assumes cross-compilers, cross-linkers, etc. are in the PATH.
# Assumes CodeSourcery sysroot is in /opt/libc.
#
# Assumes no CUDA support on target system.
#
# Since OpenCL is library based, you have to explicitly specify CPPFLAGS to
# find the OpenCL headers.  You may also need to specify LDFLAGS, depending on
# whether the OpenCL libraries are installed in a location searched by
# the linker such as /usr/lib.
#
# Does not (yet?) support MPI.
#
sh ./configure \
CPPFLAGS="-I$HOME/private/Projects/ARM/ARM-OpenCL-1.1/include" \
LDFLAGS="-L$HOME/private/Projects/ARM/ARM-OpenCL-1.1/lib -Wl,-rpath=/opt/libc/lib:/opt/libc/usr/lib -Wl,--dynamic-linker=/opt/libc/lib/ld-linux.so.3" \
--host=arm-none-linux-gnueabi \
--with-opencl \
--without-cuda \
--without-mpi


