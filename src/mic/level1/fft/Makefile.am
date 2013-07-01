#if BUILD_MPI
#    MAYBE_MPI = epmpi tpmpi
#endif
#SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(MIC_OFFLOAD_LDFLAGS)
AM_CXXFLAGS = $(MIC_OFFLOAD_CXXFLAGS)

# What is the destination for programs built from this directory?
serialmicdir = $(bindir)/Serial/MIC

# What programs should be installed to that destination?
serialmic_PROGRAMS = FFT

# How to build those programs?
FFT_SOURCES = main.cpp FFT.cpp
FFT_LDADD = $(MIC_LIBS) $(LIBS)

