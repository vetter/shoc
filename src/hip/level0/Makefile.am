if BUILD_MPI
    MAYBE_MPI = epmpi
endif

SUBDIRS	= $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../common
VPATH = $(srcdir):$(srcdir)/../../common
VPATH = $(srcdir):$(srcdir)/epmpi

AM_LDFLAGS = $(HIP_LDFLAGS)
AM_CPPFLAGS = $(HIP_INC) -Wdeprecated-register

# What is the destination for programs built from this directory?
serialhipdir = $(bindir)/Serial/HIP

# What programs should be installed in the bin/Serial/CUDA destination?
serialhip_PROGRAMS = BusSpeedDownload \
						BusSpeedReadback \
						DeviceMemory \
						MaxFlops

BusSpeedDownload_SOURCES	= main.cpp
BusSpeedDownload_LDADD		= BusSpeedDownload.o $(HIP_LIBS) $(LIBS)

BusSpeedReadback_SOURCES	= main.cpp
BusSpeedReadback_LDADD		= BusSpeedReadback.o $(HIP_LIBS) $(LIBS)

DeviceMemory_SOURCES	= main.cpp
DeviceMemory_LDADD		= DeviceMemory.o $(HIP_LIBS) $(LIBS)

MaxFlops_SOURCES	= main.cpp
MaxFlops_LDADD		= MaxFlops.o $(HIP_LIBS) $(LIBS)

