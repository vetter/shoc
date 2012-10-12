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
epopencl_PROGRAMS = FFT

# how to build those programs
FFT_SOURCES = FFT.cpp \
				fftlib.cpp \
				fft_cl.cpp
FFT_LDADD   = main_mpi.o $(L1LIBS)

