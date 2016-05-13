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
epcudadir = $(bindir)/EP/HIP

# What programs should be installed to that destination?
epcuda_PROGRAMS = Spmv

# How to build those programs?
Spmv_SOURCES = main.cpp
Spmv_LDADD = Spmv.o $(HIP_LIBS) $(LIBS)

