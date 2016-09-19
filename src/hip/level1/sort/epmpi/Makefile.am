include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# Which compiler to use to build and link
CXX = ${HIPCC}
CXXLD = ${HIPCC}

AM_LDFLAGS = $(HIP_LDFLAGS) -L/usr/lib/openmpi/lib -lmpi
AM_CPPFLAGS = $(HIP_INC) $(MPI_CPPFLAGS) -I$(top_srcdir)/src/mpi/common -I/usr/lib/openmpi/include

# What is the destination for programs built from this directory?
ephipdir = $(bindir)/EP/HIP

# What programs should be installed to that destination?
ephip_PROGRAMS = Sort

# How to build those programs?
Sort_SOURCES = main.cpp
Sort_LDADD = Sort.o sort_kernel.o $(HIP_LIBS) $(LIBS)

