include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/..:$(top_srcdir)/src/cuda/level0

# Which compiler to use?
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(CUDA_LDFLAGS) -L$(top_builddir)/src/mpi/common
AM_CPPFLAGS = $(CUDA_INC) $(MPI_CPPFLAGS) -I$(top_srcdir)/src/mpi/common

# What is the destination for programs built from this directory?
epcudadir = $(bindir)/EP/CUDA

# What programs should be installed to that destination?
epcuda_PROGRAMS = BusCont

# How to build those programs?
BusCont_SOURCES = CUDADriver.cpp \
					BusCont.cpp \
					bcmain.cpp
BusCont_LDADD = $(top_builddir)/src/cuda/level0/BusSpeedDownload.o -lSHOCCommonMPI $(CUDA_LIBS) $(LIBS)

