if BUILD_MPI
    MAYBE_MPI = epmpi
endif

SUBDIRS	= $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../common

AM_LDFLAGS = $(OPENACC_LDFLAGS)
AM_CFLAGS = $(OPENACC_CFLAGS)

# What is the destination for programs built from this directory?
serialopenaccdir = $(bindir)/Serial/OpenACC

# What programs should be installed in the bin/Serial/CUDA destination?
serialopenacc_PROGRAMS = BusSpeedDownload \
						BusSpeedReadback \
						DeviceMemory \
						MaxFlops

BusSpeedDownload_SOURCES	= main.cpp BusSpeedDownload.cpp BusSpeedFuncs.c
BusSpeedDownload_LDADD		= -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

BusSpeedReadback_SOURCES	= main.cpp BusSpeedReadback.cpp BusSpeedFuncs.c
BusSpeedReadback_LDADD		= -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

DeviceMemory_SOURCES	= main.cpp DeviceMemory.cpp DeviceMemFuncs.c
DeviceMemory_LDADD		= -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

MaxFlops_SOURCES	= main.cpp MaxFlops.cpp MaxFlopsFuncs.c
MaxFlops_LDADD		= -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

