include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# which compiler to use
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(OCL_LDFLAGS)
CPPFLAGS += $(MPI_CPPFLAGS) $(OCL_CPPFLAGS) -I${top_srcdir}/src/mpi/common
L1LIBS = $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
tpopencldir = $(bindir)/TP/OpenCL

# what programs should be installed in the destination?
tpopencl_PROGRAMS = Reduction

# how to build those programs
Reduction_SOURCES = reduction_cl.cpp \
					tpReduction.cpp \
					main.cpp
Reduction_LDADD   = $(L1LIBS)

