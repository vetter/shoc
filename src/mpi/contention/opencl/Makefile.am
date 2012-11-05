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
# Note: we would prefer to put BusSpeedDownload and main.cpp in the 
# SOURCES list.  However, our VPATH must contain src/opencl/level0 
# so that our build can find the BusSpeedDownload.cpp file.  Since
# this directory is built after that src/opencl/level0 directory is
# built, that directory already has a BusSpeedDownload.o and a main.o file.
# With our VPATH, gnumake finds those existing .o files and doesn't build
# them here.  This isn't as much a problem for BusSpeedDownload.o, which
# is built the same as for the non-contention tests, but the main.cpp file
# we use is different than the main.cpp file used in the non-contention tests.
#
# Hence, we have to list the object files for those files in our LDADD list.
#
BusCont_SOURCES = OCLDriver.cpp \
					BusCont.cpp \
					bcmain.cpp
BusCont_LDADD = $(top_builddir)/src/opencl/level0/BusSpeedDownload.o -lSHOCCommonMPI $(OCL_LIBS) $(LIBS)

