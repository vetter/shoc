SHOC_ROOT=../../..
include $(SHOC_ROOT)/config/config.mk

OBJ= \
 RandomPairs_mpi.o \
 MPI2DGridProgram_mpi.o \
 MPIHostStencil_mpi.o \
 MPIHostStencilFactory_mpi.o \
 MPIStencilUtil_mpi.o


ARCHIVE=$(SHOC_ROOT)/lib/libSHOCCommonMPI.a

all cuda opencl: $(ARCHIVE)

$(ARCHIVE): $(OBJ)
	$(AR) $(ARFLAGS) $@ $(OBJ)

clean:
	${RM} $(OBJ) $(ARCHIVE)

distclean: clean
	${RM} $(DEP) *~ .buildflags Makefile


include $(SHOC_ROOT)/config/targets.mk

