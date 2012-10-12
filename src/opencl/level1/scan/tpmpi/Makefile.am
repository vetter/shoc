include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# Which compiler to use to link
CXXLD = ${MPICXX}

AM_LDFLAGS = $(OCL_LDFLAGS)
CPPFLAGS += $(OCL_CPPFLAGS) -I${top_srcdir}/src/mpi/common
L1LIBS = -lSHOCCommonOpenCL -lSHOCCommon $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
tpopencldir = $(bindir)/TP/OpenCL

# what programs should be installed in the destination?
tpopencl_PROGRAMS = Scan

# how to build those programs
Scan_SOURCES = scan_cl.cpp
Scan_LDADD   = main_mpi.o tpScan_mpi.o $(L1LIBS)

