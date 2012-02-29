SHOC_ROOT=../../../..
include $(SHOC_ROOT)/config/config.mk

VPATH = ..

BENCHMARKOBJ_OCL= \
 MTBusCont_mpi.o 

BENCHMARKPROG_OCL = $(patsubst %_mpi.o,$(SHOC_ROOT)/bin/EP/OpenCL/%,$(BENCHMARKOBJ_OCL))

COMMONOBJ_OCL= \
 $(SHOC_ROOT)/src/opencl/level0/BusSpeedDownload_ocl.o \
 OCLDriver_ocl.o \
 main_mpi.o

ALLOBJ=$(COMMONOBJ_OCL) $(BENCHMARKOBJ_OCL)

LIBS+=-lSHOCCommonMPI -lSHOCCommon
OCL_LIBS+=-lSHOCCommonOpenCL

CPPFLAGS+=-I${SHOC_ROOT}/src/mpi/common

all: $(BENCHMARKPROG_OCL)

$(SHOC_ROOT)/bin/EP/OpenCL/%: %_mpi.o
	$(MPICXX) -o $@ $(CXXFLAGS) $(OCL_CXXFLAGS) $(LDFLAGS) $(OCL_LDFLAGS) $(COMMONOBJ_OCL) $< $(LIBS) $(OCL_LIBS)

$(BENCHMARKPROG_OCL): $(COMMONOBJ_OCL)

clean:
	${RM} $(ALLOBJ) $(BENCHMARKPROG_OCL)

distclean: clean
	${RM} $(DEP) *~ .buildflags Makefile

include $(SHOC_ROOT)/config/targets.mk
