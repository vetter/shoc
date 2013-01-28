include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../common:$(srcdir)/../../../common

# which compiler to use
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(OPENACC_LDFLAGS)
AM_CPPFLAGS = ${MPI_CPPFLAGS} -I${top_srcdir}/src/mpi/common
AM_CFLAGS	= $(OPENACC_CFLAGS)

# what is the destination for programs built from this directory?
epopenaccdir = $(bindir)/EP/OpenACC

# what programs should be installed in the destination?
epopenacc_PROGRAMS = BusSpeedDownload \
					 BusSpeedReadback \
					 MaxFlops         \
					 DeviceMemory

# how to build those programs
BusSpeedDownload_SOURCES	= main.cpp BusSpeedDownload.cpp BusSpeedFuncs.c
BusSpeedDownload_LDADD		= -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

BusSpeedReadback_SOURCES	= main.cpp BusSpeedReadback.cpp BusSpeedFuncs.c
BusSpeedReadback_LDADD		= -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

DeviceMemory_SOURCES	= main.cpp DeviceMemory.cpp DeviceMemFuncs.c
DeviceMemory_LDADD		= -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

MaxFlops_SOURCES	= main.cpp MaxFlops.cpp MaxFlopsFuncs.c
MaxFlops_LDADD		= -lSHOCCommon $(OPENACC_LIBS) $(LIBS)

