if BUILD_MPI
    MAYBE_MPI = epmpi
endif
SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(top_srcdir)/src/cuda/common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
serialcudadir = $(bindir)/Serial/CUDA

# What programs should be installed to that destination?
serialcuda_PROGRAMS = Stability

# How to build those programs?
Stability_SOURCES = Stability.cpp \
                main.cpp
Stability_LDADD = $(top_builddir)/src/cuda/level1/fft/fftlib.o $(CUDA_LIBS) -lcufft $(LIBS)

