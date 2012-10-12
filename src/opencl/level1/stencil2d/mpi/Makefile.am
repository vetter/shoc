include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# Which compiler to use to link?
CXXLD = ${MPICXX}

AM_LDFLAGS = $(OCL_LDFLAGS)
CPPFLAGS += $(OCL_CPPFLAGS) -I${top_srcdir}/src/mpi/common
L1LIBS = -lSHOCCommonOpenCL -lSHOCCommon $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
tpopencldir = $(bindir)/TP/OpenCL

# what programs should be installed in the destination?
tpopencl_PROGRAMS = Stencil2D

# how to build those programs
Stencil2D_SOURCES = OpenCLStencil.cpp \
				OpenCLStencilFactory.cpp \
				CommonOpenCLStencilFactory.cpp \
				stencil2d_cl.cpp
Stencil2D_LDADD   = main_mpi.o Stencil2Dmain_mpi.o MPIOpenCLStencil_mpi.o MPIOpenCLStencilFactory_mpi.o $(L1LIBS)

