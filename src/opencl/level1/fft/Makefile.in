SHOC_ROOT=../../../..
include $(SHOC_ROOT)/config/config.mk

VPATH:=../../common

LDFLAGS+=$(OCL_LDFLAGS)
CPPFLAGS+=$(OCL_CPPFLAGS) -I${SHOC_ROOT}/src/mpi/common
LIBS:=-lSHOCCommonOpenCL -lSHOCCommon $(OCL_LIBS) $(LIBS)

BENCHMARKOBJ=  \
 FFT.o \

BENCHMARKPROG = $(patsubst %.o,$(SHOC_ROOT)/bin/Serial/OpenCL/%,$(BENCHMARKOBJ))
BENCHMARKPROG_MPI = $(patsubst %.o,$(SHOC_ROOT)/bin/EP/OpenCL/%,$(BENCHMARKOBJ))

COMMONOBJ= \
 main.o fftlib.o fft_cl.o

COMMONOBJ_MPI= \
 main_mpi.o fftlib.o fft_cl.o

LIBDEP=$(SHOC_ROOT)/lib/libSHOCCommonOpenCL.a $(SHOC_ROOT)/lib/libSHOCCommon.a 

ALLOBJ=$(COMMONOBJ) $(BENCHMARKOBJ)
TARGETS=${BENCHMARKPROG}

ifeq ($(USE_MPI),yes)
ALLOBJ+=${COMMONOBJ_MPI}
TARGETS+=${BENCHMARKPROG_MPI}
endif

all: ${TARGETS}

$(SHOC_ROOT)/bin/Serial/OpenCL/%: %.o  $(LIBDEP)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(COMMONOBJ) $< $(LIBS)

$(SHOC_ROOT)/bin/EP/OpenCL/% : %.o  $(LIBDEP)
	$(MPICXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(COMMONOBJ_MPI) $< $(LIBS)

$(BENCHMARKPROG): $(COMMONOBJ)

$(BENCHMARKPROG_MPI): $(COMMONOBJ_MPI)


clean:
	${RM} *_cl.cpp $(ALLOBJ) $(BENCHMARKPROG) $(BENCHMARKPROG_MPI)

distclean: clean
	${RM} $(DEP) *~ .buildflags Makefile

include $(SHOC_ROOT)/config/targets.mk
