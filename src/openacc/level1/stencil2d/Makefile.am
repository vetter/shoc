#if BUILD_MPI
#    MAYBE_MPI = tpmpi
#endif
#SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(OPENACC_LDFLAGS)
AM_CFLAGS	= $(OPENACC_CFLAGS)

# what is the destination for programs built from this directory?
openaccdir = $(bindir)/Serial/OpenACC

# what programs should be installed in the destination?
openacc_PROGRAMS = Stencil2D

# how to build those programs
Stencil2D_SOURCES = Stencil2Dmain.cpp \
				StencilFuncs.c \
                main.cpp
Stencil2D_LDADD   = -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

