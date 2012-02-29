SHOC_ROOT=../../..
include $(SHOC_ROOT)/config/config.mk

VPATH := ../common

LDFLAGS+=$(CUDA_LDFLAGS)
LIBS+=$(CUDA_LIBS)
CPPFLAGS+=$(CUDA_INC)


BENCHMARKOBJ= \
 BusSpeedDownload.o \
 BusSpeedReadback.o \
 DeviceMemory.o \
 MaxFlops.o


BENCHMARKPROG = $(patsubst %.o,$(SHOC_ROOT)/bin/Serial/CUDA/%,$(BENCHMARKOBJ))
BENCHMARKPROG_MPI = $(patsubst %.o,$(SHOC_ROOT)/bin/EP/CUDA/%,$(BENCHMARKOBJ))

COMMONOBJ= \
 main.o

COMMONOBJ_MPI= \
 main_mpi.o

LIBS+=-lSHOCCommon
LIBDEP=$(SHOC_ROOT)/lib/libSHOCCommon.a 

ALLOBJ=$(COMMONOBJ) $(BENCHMARKOBJ)
TARGETS=${BENCHMARKPROG}

ifeq ($(USE_MPI),yes)
ALLOBJ+=${COMMONOBJ_MPI}
TARGETS+=${BENCHMARKPROG_MPI}
CPPFLAGS+=-I${SHOC_ROOT}/src/mpi/common
endif

all: ${TARGETS}

$(SHOC_ROOT)/bin/Serial/CUDA/%: %.o  $(LIBDEP)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(COMMONOBJ) $< $(LIBS)

$(SHOC_ROOT)/bin/EP/CUDA/%: %.o  $(LIBDEP)
	$(MPICXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(COMMONOBJ_MPI) $< $(LIBS)

$(BENCHMARKPROG): $(COMMONOBJ)

$(BENCHMARKPROG_MPI): $(COMMONOBJ_MPI)

clean:
	${RM} $(ALLOBJ) $(BENCHMARKPROG) $(BENCHMARKPROG_MPI)

distclean: clean
	${RM} $(DEP) *~ .buildflags Makefile

include $(SHOC_ROOT)/config/targets.mk
