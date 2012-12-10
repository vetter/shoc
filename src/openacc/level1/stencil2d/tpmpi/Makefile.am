include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# which compiler to use
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(OPENACC_LDFLAGS)
AM_CPPFLAGS	= ${MPI_CPPFLAGS} -I${top_srcdir}/src/mpi/common
AM_CFLAGS	= $(OPENACC_CFLAGS)

# what is the destination for programs built from this directory?
tpopenaccdir = $(bindir)/TP/OpenACC

# what programs should be installed in the destination?
tpopenacc_PROGRAMS = Stencil2D

# how to build those programs
Stencil2D_SOURCES = Stencil2Dmain.cpp \
				StencilFuncs.c \
                main.cpp
Stencil2D_LDADD   = -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

