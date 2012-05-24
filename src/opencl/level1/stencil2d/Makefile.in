SHOC_ROOT=../../../..
include ${SHOC_ROOT}/config/config.mk

VPATH:=../../common

LDFLAGS     += ${OCL_LDFLAGS}
CPPFLAGS    += ${OCL_CPPFLAGS} -I${SHOC_ROOT}/src/common 
LIBS        := -lSHOCCommonOpenCL -lSHOCCommon $(OCL_LIBS) $(LIBS)

TARGET      = ${SHOC_ROOT}/bin/Serial/OpenCL/Stencil2D
MPITARGET   = ${SHOC_ROOT}/bin/TP/OpenCL/Stencil2D
COMMON_SRCS = OpenCLStencil.cpp \
                stencil2d_cl.cpp \
                CommonOpenCLStencilFactory.cpp
OBJS        = main.o Stencil2Dmain.o OpenCLStencilFactory.o ${COMMON_SRCS:.cpp=.o}
MPI_OBJS    = main_mpi.o Stencil2Dmain_mpi.o MPIOpenCLStencil_mpi.o MPIOpenCLStencilFactory_mpi.o ${COMMON_SRCS:.cpp=.o}

LIBDEP_SER=$(SHOC_ROOT)/lib/libSHOCCommonOpenCL.a $(SHOC_ROOT)/lib/libSHOCCommon.a 
LIBDEP_PAR=$(LIBDEP_SER) $(SHOC_ROOT)/lib/libSHOCCommonMPI.a 

TARGETS=${TARGET}
ifeq ($(USE_MPI),yes)
TARGETS+=${MPITARGET}
CPPFLAGS    += -I${SHOC_ROOT}/src/mpi/common
MPI_LIBS    += -lSHOCCommonMPI
endif

all: ${TARGETS}

${TARGET}: ${OBJS}  $(LIBDEP_SER)
	${CXX} -o $@ ${CXXFLAGS} ${LDFLAGS} ${OBJS} ${LIBS}


$(SHOC_ROOT)/lib/libSHOCCommonMPI.a:
	${MAKE} -C ${SHOC_ROOT}/src/mpi/common

${MPITARGET}:   ${MPI_OBJS}  $(LIBDEP_PAR)
	${MPICXX} -o $@ ${CXXFLAGS} ${LDFLAGS} ${MPI_OBJS} ${MPI_LIBS} ${LIBS}


clean:
	${RM} ${OBJS} ${MPI_OBJS} ${TARGET} ${MPITARGET}

distclean: clean
	${RM} ${DEPS} .buildflags *_cl.cpp Makefile

include ${SHOC_ROOT}/config/targets.mk

