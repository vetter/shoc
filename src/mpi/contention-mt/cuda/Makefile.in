SHOC_ROOT=../../../..
include $(SHOC_ROOT)/config/config.mk

VPATH = ..

BENCHMARKOBJ_CUDA= \
 MTBusCont_mpi.o 

BENCHMARKPROG_CUDA = $(patsubst %_mpi.o,$(SHOC_ROOT)/bin/EP/CUDA/%,$(BENCHMARKOBJ_CUDA))

COMMONOBJ_CUDA= \
 $(SHOC_ROOT)/src/cuda/level0/BusSpeedDownload_cuda.o \
 CUDADriver_cuda.o\
 main_mpi.o

ALLOBJ=$(COMMONOBJ_CUDA) $(BENCHMARKOBJ_CUDA)

LIBS+=-lSHOCCommonMPI -lSHOCCommon
#CUDA_LIBS+=-lSHOCCommonCUDA

CPPFLAGS+=-I${SHOC_ROOT}/src/mpi/common

MPI_LIBS    = -lSHOCCommonMPI

all: $(BENCHMARKPROG_CUDA)

$(SHOC_ROOT)/bin/EP/CUDA/%: %_mpi.o
	$(MPICXX) -o $@ $(CXXFLAGS) $(CUDA_CPPFLAGS) $(CUDA_LDFLAGS) $(COMMONOBJ_CUDA) $< $(LIBS) ${MPI_LIBS} $(CUDA_LIBS)

$(BENCHMARKPROG_CUDA): $(COMMONOBJ_CUDA)

clean:
	${RM} $(ALLOBJ) $(BENCHMARKPROG_CUDA)

distclean: clean
	${RM} $(DEP) *~ .buildflags Makefile

include $(SHOC_ROOT)/config/targets.mk
