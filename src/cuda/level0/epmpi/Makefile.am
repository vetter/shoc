include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../common

# Which compiler to use to link
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC) $(MPI_CPPFLAGS) -I$(top_srcdir)/src/mpi/common

# What is the destination for programs built from this directory?
epcudadir = $(bindir)/EP/CUDA

# What programs should be installed in the destination?
epcuda_PROGRAMS = BusSpeedDownload \
						BusSpeedReadback \
						DeviceMemory \
						MaxFlops

BusSpeedDownload_SOURCES	= main.cpp
BusSpeedDownload_LDADD		= BusSpeedDownload.o $(CUDA_LIBS) $(LIBS)

BusSpeedReadback_SOURCES	= main.cpp
BusSpeedReadback_LDADD		= BusSpeedReadback.o $(CUDA_LIBS) $(LIBS)

DeviceMemory_SOURCES	= main.cpp
DeviceMemory_LDADD		= DeviceMemory.o $(CUDA_LIBS) $(LIBS)

MaxFlops_SOURCES	= main.cpp
MaxFlops_LDADD		= MaxFlops.o $(CUDA_LIBS) $(LIBS)

