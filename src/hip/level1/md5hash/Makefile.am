if BUILD_MPI
    MAYBE_MPI = epmpi
endif
SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
serialcudadir = $(bindir)/Serial/CUDA

# What programs should be installed to that destination?
serialcuda_PROGRAMS = MD5Hash

# How to build those programs?
MD5Hash_SOURCES = main.cpp
MD5Hash_LDADD = MD5Hash.o $(CUDA_LIBS) $(LIBS)

