include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../common

# which compiler to use to link
CXX= ${MPICXX}
CXXLD = ${MPICXX}

# which flags to use
AM_LDFLAGS = $(OCL_LDFLAGS)
CPPFLAGS += $(MPI_CPPFLAGS) $(OCL_CPPFLAGS) -I$(top_srcdir)/src/opencl/common -I$(top_srcdir)/src/mpi/common
L0LIBS = -lSHOCCommonOpenCL -lSHOCCommon $(OCL_LIBS) $(LIBS)

# what is the destination for programs built from this directory?
epopencldir = $(bindir)/EP/OpenCL

# what programs should be installed in the bin/Serial/OpenCL destination?
epopencl_PROGRAMS = BusSpeedDownload \
						BusSpeedReadback \
						DeviceMemory \
						KernelCompile \
						MaxFlops \
						QueueDelay

BusSpeedDownload_SOURCES	= BusSpeedDownload.cpp main.cpp
BusSpeedDownload_LDADD		= $(L0LIBS)

BusSpeedReadback_SOURCES	= BusSpeedReadback.cpp main.cpp
BusSpeedReadback_LDADD		= $(L0LIBS)

DeviceMemory_SOURCES	= DeviceMemory.cpp main.cpp
DeviceMemory_LDADD		= $(L0LIBS)

KernelCompile_SOURCES	= KernelCompile.cpp main.cpp
KernelCompile_LDADD		= $(L0LIBS)

MaxFlops_SOURCES	= MaxFlops.cpp main.cpp
MaxFlops_LDADD		= $(L0LIBS)

QueueDelay_SOURCES	= QueueDelay.cpp main.cpp
QueueDelay_LDADD	= $(L0LIBS)


