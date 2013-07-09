#if BUILD_MPI
#    MAYBE_MPI = tpmpi
#endif
#SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

MIC_ALTOFFLOAD_FLAGS=-opt-assume-safe-padding -no-opt-prefetch -opt-threads-per-core=4 -restrict -opt-streaming-stores always -opt-streaming-cache-evict=0

AM_CXXFLAGS = $(MIC_OFFLOAD_CXXFLAGS) $(MIC_ALTOFFLOAD_FLAGS)
AM_LDFLAGS = $(MIC_OFFLOAD_LDFLAGS) $(MIC_ALTOFFLOAD_FLAGS)

# What is the destination for programs built from this directory?
serialmicdir = $(bindir)/Serial/MIC

# What programs should be installed to that destination?
serialmic_PROGRAMS = Stencil2D

# How to build those programs?
Stencil2D_SOURCES = MICStencil.cpp \
				CommonMICStencilFactory.cpp \
				Stencil2Dmain.cpp \
				MICStencilFactory.cpp \
				MICStencilKernel.cpp \
                main.cpp
Stencil2D_LDADD = $(MIC_LIBS) $(LIBS)

