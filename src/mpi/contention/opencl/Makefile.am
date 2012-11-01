include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/..:$(top_srcdir)/src/opencl/level0

# Which compiler to use?
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(OCL_LDFLAGS) -L$(top_builddir)/src/mpi/common
AM_CPPFLAGS = $(OCL_CPPFLAGS) $(MPI_CPPFLAGS) -I$(top_srcdir)/src/mpi/common

# What is the destination for programs built from this directory?
epopencldir = $(bindir)/EP/OpenCL

# What programs should be installed to that destination?
epopencl_PROGRAMS = BusCont

# How to build those programs?
BusCont_SOURCES = OCLDriver.cpp \
					BusCont.cpp \
					BusSpeedDownload.cpp \
					main.cpp
BusCont_LDADD = -lSHOCCommonMPI $(OCL_LIBS) $(LIBS)

