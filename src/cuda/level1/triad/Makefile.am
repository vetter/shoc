SHOC_ROOT=../../../..
include $(SHOC_ROOT)/config/config.mk

VPATH := ../../common

LDFLAGS+=$(CUDA_LDFLAGS)
CPPFLAGS+=$(CUDA_INC) -I../../include

BENCHMARKOBJ= \
 Triad.o \

BENCHMARKPROG = $(patsubst %.o,$(SHOC_ROOT)/bin/Serial/CUDA/%,$(BENCHMARKOBJ))
BENCHMARKPROG_MPI = $(patsubst %.o,$(SHOC_ROOT)/bin/EP/CUDA/%,$(BENCHMARKOBJ))

COMMONOBJ= \
 main.o

COMMONOBJ_MPI= \
 main_mpi.o

LIBS:= -lSHOCCommon $(LIBS)

ALLOBJ=$(COMMONOBJ) $(BENCHMARKOBJ)
TARGETS=${BENCHMARKPROG}

ifeq ($(USE_MPI),yes)
ALLOBJ+=${COMMONOBJ_MPI}
TARGETS+=${BENCHMARKPROG_MPI}
CPPFLAGS+=-I${SHOC_ROOT}/src/mpi/common
endif

all: ${TARGETS}

$(SHOC_ROOT)/bin/Serial/CUDA/%: %.o
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(COMMONOBJ) $< $(LIBS) $(CUDA_LIBS)

$(SHOC_ROOT)/bin/EP/CUDA/%: %.o
	$(MPICXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(COMMONOBJ_MPI) $< $(LIBS) $(CUDA_LIBS)

$(BENCHMARKPROG): $(COMMONOBJ)

$(BENCHMARKPROG_MPI): $(COMMONOBJ_MPI)

clean:
	${RM} $(ALLOBJ) $(BENCHMARKPROG) $(BENCHMARKPROG_MPI)

distclean: clean
	${RM} $(DEP) *~ .buildflags Makefile

include $(SHOC_ROOT)/config/targets.mk
