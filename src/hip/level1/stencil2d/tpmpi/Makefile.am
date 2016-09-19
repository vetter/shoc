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
tpcudadir = $(bindir)/TP/HIP

# What programs should be installed to that destination?
tpcuda_PROGRAMS = Stencil2D

# How to build those programs?
Stencil2D_SOURCES = CUDAStencil.cpp \
				CommonCUDAStencilFactory.cpp \
				Stencil2Dmain.cpp \
				CUDAStencilFactory.cpp \
                main.cpp
Stencil2D_LDADD = CUDAStencilKernel.o $(HIP_LIBS) $(LIBS)

