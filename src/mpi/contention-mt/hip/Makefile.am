include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/..:$(top_srcdir)/src/hip/level0

# Which compiler to use?
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(HIP_LDFLAGS) -L$(top_builddir)/src/mpi/common
AM_CPPFLAGS = $(HIP_INC) $(MPI_CPPFLAGS) -I$(top_srcdir)/src/mpi/common

# What is the destination for programs built from this directory?
ephipdir = $(bindir)/EP/HIP

# What programs should be installed to that destination?
ephip_PROGRAMS = MTBusCont

# How to build those programs?
MTBusCont_SOURCES = CUDADriver.cpp \
					MTBusCont.cpp \
					mtbcmain.cpp
MTBusCont_LDADD = $(top_builddir)/src/hip/level0/BusSpeedDownload.o -lSHOCCommonMPI $(HIP_LIBS) $(LIBS)

