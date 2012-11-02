include ${top_builddir}/config/common.mk
include ${top_builddir}/config/targets.mk

# Which compiler to use to build and link?
CXX = ${MPICXX}
CXXLD = ${MPICXX}

noinst_LIBRARIES    = libSHOCCommonMPI.a
libSHOCCommonMPI_a_SOURCES = RandomPairs.cpp \
	MPI2DGridProgram.cpp \
	MPIHostStencil.cpp \
	MPIHostStencilFactory.cpp \
	MPIStencilUtil.cpp

