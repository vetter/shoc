#!/bin/sh

# do the actual configuration
# if using TAUcuda:
#sh ./configure CC=tau_cc.sh CXX=tau_cxx.sh \

# otherwise:
sh ./configure \
CPPFLAGS="-I/sw/keeneland/cuda/4.1/linux_binary/include"

