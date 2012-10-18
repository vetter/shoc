if BUILD_MPI
    MAYBE_MPI = tpmpi
endif
SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(OCL_LDFLAGS)
CPPFLAGS += $(OCL_CPPFLAGS)
L1LIBS = $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
serialopencldir = $(bindir)/Serial/OpenCL

# what programs should be installed in the destination?
serialopencl_PROGRAMS = Stencil2D

# how to build those programs
Stencil2D_SOURCES = Stencil2Dmain.cpp \
				OpenCLStencil.cpp \
				OpenCLStencilFactory.cpp \
				CommonOpenCLStencilFactory.cpp \
				stencil2d_cl.cpp \
                main.cpp
Stencil2D_LDADD   = $(L1LIBS)

