#if BUILD_MPI
#    MAYBE_MPI = epmpi
#endif

#SUBDIRS	= $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../common

AM_LDFLAGS = $(MIC_OFFLOAD_LDFLAGS)
AM_CXXFLAGS = $(MIC_OFFLOAD_CXXFLAGS)


# What is the destination for programs built from this directory?
serialmicdir = $(bindir)/Serial/MIC

# What programs should be installed in the bin/Serial/MIC destination?
serialmic_PROGRAMS = BusSpeedDownload \
						BusSpeedReadback \
						DeviceMemory \
						MaxFlops

BusSpeedDownload_SOURCES	= main.cpp BusSpeedDownload.cpp
BusSpeedDownload_LDADD		= $(MIC_LIBS) $(LIBS)

BusSpeedReadback_SOURCES	= main.cpp BusSpeedReadback.cpp
BusSpeedReadback_LDADD		= $(MIC_LIBS) $(LIBS)

DeviceMemory_SOURCES	= main.cpp DeviceMemory.cpp
DeviceMemory_LDADD		= $(MIC_LIBS) $(LIBS)

MaxFlops_SOURCES	= main.cpp MaxFlops.cpp
MaxFlops_LDADD		= $(MIC_LIBS) $(LIBS)

