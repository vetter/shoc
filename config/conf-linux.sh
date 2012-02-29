#!/bin/sh


# do the actual configuration
#
# the configure script looks for CUDA using the PATH, but since OpenCL
# is library based, you have to explicitly specify CPPFLAGS to find
# the OpenCL headers.  You may also need to specify LDFLAGS, depending on
# whether the OpenCL libraries are installed in a location searched by
# the linker such as /usr/lib.
#
sh ./configure \
CPPFLAGS="-I/usr/local/cuda/include"


