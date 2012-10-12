include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# which compiler to use to link
CXXLD = ${MPICXX}

AM_LDFLAGS = $(OCL_LDFLAGS)
CPPFLAGS += $(OCL_CPPFLAGS) -I${top_srcdir}/src/mpi/common
L1LIBS = -lSHOCCommonOpenCL -lSHOCCommon $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
epopencldir = $(bindir)/EP/OpenCL

# what programs should be installed in the destination?
epopencl_PROGRAMS = SGEMM

# how to build those programs
SGEMM_SOURCES = SGEMM.cpp \
				sgemmN_cl.cpp
SGEMM_LDADD   = main_mpi.o $(L1LIBS)

