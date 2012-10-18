if BUILD_MPI
    MAYBE_MPI = epmpi
endif

SUBDIRS	= $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../common

AM_LDFLAGS = $(OCL_LDFLAGS)
AM_CPPFLAGS = $(OCL_CPPFLAGS)

# what is the destination for programs built from this directory?
serialopencldir = $(bindir)/Serial/OpenCL

# what programs should be installed in the bin/Serial/OpenCL destination?
serialopencl_PROGRAMS = BusSpeedDownload \
						BusSpeedReadback \
						DeviceMemory \
						KernelCompile \
						MaxFlops \
						QueueDelay

BusSpeedDownload_SOURCES	= BusSpeedDownload.cpp main.cpp
BusSpeedDownload_LDADD		= $(OCL_LIBS) $(LIBS)

BusSpeedReadback_SOURCES	= BusSpeedReadback.cpp main.cpp
BusSpeedReadback_LDADD		= $(OCL_LIBS) $(LIBS)

DeviceMemory_SOURCES	= DeviceMemory.cpp main.cpp
DeviceMemory_LDADD		= $(OCL_LIBS) $(LIBS)

KernelCompile_SOURCES	= KernelCompile.cpp main.cpp
KernelCompile_LDADD		= $(OCL_LIBS) $(LIBS)

MaxFlops_SOURCES	= MaxFlops.cpp main.cpp
MaxFlops_LDADD		= $(OCL_LIBS) $(LIBS)

QueueDelay_SOURCES	= QueueDelay.cpp main.cpp
QueueDelay_LDADD	= $(OCL_LIBS) $(LIBS)


