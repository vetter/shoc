include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# which compiler to use
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(OCL_LDFLAGS)
CPPFLAGS += $(MPI_CPPFLAGS) $(OCL_CPPFLAGS) -I${top_srcdir}/src/mpi/common
L1LIBS = $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
tpopencldir = $(bindir)/TP/OpenCL

# what programs should be installed in the destination?
tpopencl_PROGRAMS = Stencil2D

# how to build those programs
Stencil2D_SOURCES = OpenCLStencil.cpp \
				OpenCLStencilFactory.cpp \
				CommonOpenCLStencilFactory.cpp \
				stencil2d_cl.cpp \
				Stencil2Dmain.cpp \
				MPIOpenCLStencil.cpp \
				MPIOpenCLStencilFactory.cpp \
				main.cpp
Stencil2D_LDADD   = $(L1LIBS)

