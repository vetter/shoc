include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(top_srcdir)/src/cuda/common

# Which compiler to use to build and link
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC) $(MPI_CPPFLAGS) -I$(top_srcdir)/src/mpi/common

# What is the destination for programs built from this directory?
epcudadir = $(bindir)/EP/CUDA

# What programs should be installed to that destination?
epcuda_PROGRAMS = Stability_mpi

# How to build those programs?
Stability_mpi_SOURCES = Stability.cpp \
                main.cpp
Stability_mpi_LDADD = $(top_builddir)/src/cuda/level1/fft/fftlib.o $(CUDA_LIBS) -lcufft $(LIBS)

