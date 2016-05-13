include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../common

# Which compiler to use to link
CXX = ${HIPCC}
CXXLD = ${HIPCC}

AM_LDFLAGS = $(HIP_LDFLAGS)
AM_CPPFLAGS = $(HIP_INC) $(MPI_CPPFLAGS) -I$(top_srcdir)/src/mpi/common -D__HIP_PLATFORM_HCC__= -I/opt/hcc/include -I/usr/lib/openmpi/include -Wdeprecated-register 

# What is the destination for programs built from this directory?
epcudadir = $(bindir)/EP/HIP

# What programs should be installed in the destination?
epcuda_PROGRAMS = BusSpeedDownload \
						BusSpeedReadback \
						DeviceMemory \
						MaxFlops

BusSpeedDownload_SOURCES	= main.cpp
BusSpeedDownload_LDADD		= BusSpeedDownload.o $(HIP_PATH)/src/hip_hcc.o -lstdc++ -lc++ -lSHOCCommon -L/opt/hcc/lib -L/opt/hsa/lib -lhc_am -lhsa-runtime64 -ldl -lpthread -lmcwamp -lm -L/usr/lib/openmpi/lib -lmpi -lhwloc $(HIP_LIBS) $(LIBS) 

BusSpeedReadback_SOURCES	= main.cpp
BusSpeedReadback_LDADD		= BusSpeedReadback.o /home/aditya/Documents/HIP-privatestaging/src/hip_hcc.o -lstdc++ -lc++ -lSHOCCommon -L/opt/hcc/lib -L/opt/hsa/lib -lhc_am -lhsa-runtime64 -ldl -lpthread -lmcwamp -lm -L/usr/lib/openmpi/lib -lmpi -lhwloc $(HIP_LIBS) $(LIBS) 

DeviceMemory_SOURCES	= main.cpp
DeviceMemory_LDADD		= DeviceMemory.o /home/aditya/Documents/HIP-privatestaging/src/hip_hcc.o -lstdc++ -lc++ -L/opt/hcc/lib -L/opt/hsa/lib -lhc_am -lhsa-runtime64 -ldl -lpthread -lmcwamp -lm -lSHOCCommon -L/usr/lib/openmpi/lib -lmpi -lhwloc $(HIP_LIBS) $(LIBS) 

MaxFlops_SOURCES	= main.cpp
MaxFlops_LDADD		= MaxFlops.o /home/aditya/Documents/HIP-privatestaging/src/hip_hcc.o -lstdc++ -lc++ -L/opt/hcc/lib -L/opt/hsa/lib -lhc_am -lhsa-runtime64 -ldl -lpthread -lmcwamp -lm -lSHOCCommon -L/usr/lib/openmpi/lib -lmpi -lhwloc $(HIP_LIBS) $(LIBS) 

