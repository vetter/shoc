SHOC_ROOT=../../../..
include $(SHOC_ROOT)/config/config.mk

VPATH := ../../common

LDFLAGS+=$(CUDA_LDFLAGS)
CPPFLAGS+=$(CUDA_INC) -I../../include
LIBS:= -lSHOCCommon $(LIBS)
LIBDEP=$(SHOC_ROOT)/lib/libSHOCCommon.a 

TARGET     = ${SHOC_ROOT}/bin/Serial/CUDA/Reduction
EPTARGET   = ${SHOC_ROOT}/bin/EP/CUDA/Reduction
TPTARGET   = ${SHOC_ROOT}/bin/TP/CUDA/Reduction

TARGETS=$(TARGET) 
ifeq ($(USE_MPI),yes)
TARGETS += ${EPTARGET} $(TPTARGET)

CPPFLAGS+=-I${SHOC_ROOT}/src/mpi/common
MPI_LIBS    = -lSHOCCommonMPI
endif

OBJS     = Reduction.o main.o
EPOBJS   = Reduction.o main_mpi.o
TPOBJS   = tpRedLaunchKernel.o tpReduction_mpi.o main_mpi.o

LIBDEP_PAR=$(LIBDEP_SER) $(SHOC_ROOT)/lib/libSHOCCommonMPI.a 

all: ${TARGETS}

${TARGET}: ${OBJS}
	${CXX} -o $@ ${CXXFLAGS} ${LDFLAGS} ${OBJS} ${LIBS} ${CUDA_LIBS}

${EPTARGET}: ${EPOBJS} $(LIBDEP_PAR)
	${MPICXX} -o $@ ${CXXFLAGS} ${LDFLAGS} ${EPOBJS} ${LIBS} ${MPI_LIBS} ${CUDA_LIBS}

${TPTARGET}: ${TPOBJS} $(LIBDEP_PAR)
	${MPICXX} -o $@ ${CXXFLAGS} ${LDFLAGS} ${TPOBJS} ${LIBS} ${MPI_LIBS} ${CUDA_LIBS}

$(SHOC_ROOT)/lib/libSHOCCommonMPI.a:
	${MAKE} -C ${SHOC_ROOT}/src/mpi/common

clean:
	${RM} ${OBJS} ${EPOBJS} $(TPOBJS) ${TARGET} ${EPTARGET} $(TPTARGET)

distclean: clean
	${RM} $(DEP) *~ .buildflags Makefile

include $(SHOC_ROOT)/config/targets.mk
