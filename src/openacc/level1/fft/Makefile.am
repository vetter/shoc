if BUILD_MPI
    MAYBE_MPI = epmpi
endif
SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(OPENACC_LDFLAGS)
AM_CFLAGS	= $(OPENACC_CFLAGS)

# what is the destination for programs built from this directory?
serialopenaccdir = $(bindir)/Serial/OpenACC

# what programs should be installed in the destination?
serialopenacc_PROGRAMS = FFT

# how to build those programs
FFT_SOURCES = FFT.cpp \
				FFTFuncs.c \
                main.cpp
FFT_LDADD   = -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

