## Lens
#
## === OpenCL Configuration ===
#
## Pointer to the root of the OpenCL installation.
#OPENCL=/usr/lib64
#OCL_CPPFLAGS= -I$(SHOC_ROOT)/src/opencl/common
#
#OCL_CPPFLAGS+=-I/usr/include/CL
#OCL_CPPFLAGS+=-I$(OPENCL)/OpenCL/common/inc -I$(OPENCL)/shared/inc
#OCL_LDFLAGS=
#OCL_LIBS=-lOpenCL
#
## === CUDA Configuration ===
#
#CUDA=/sw/analysis-x64/cuda/3.0b/sl5.0_binary
#
#NVCC?=$(CUDA)/bin/nvcc
#CUDA_CXX?=$(NVCC)
#CUDA_LD?=$(NVCC)
#CUDA_INC?=-I$(CUDA)/include
#
## Set the rpath to the CUDA lib folder
#CUDA_RPATH=${CUDA}/lib64
#CUDA_LDFLAGS=--linker-options -rpath=$(CUDA_RPATH)
#
## Compilation flags for CUDA
#CUDA_CPPFLAGS=-I${CUDA}/include
#CUDA_CPPFLAGS+=-I${SHOC_ROOT}/src/cuda/include
## Go ahead and generate all necessary CUDA binaries/ptx
#CUDA_CPPFLAGS+= -gencode=arch=compute_10,code=sm_10 # binary for G80  
#CUDA_CPPFLAGS+= -gencode=arch=compute_13,code=sm_13 # binary for Tesla
##CUDA_CPPFLAGS+= -gencode=arch=compute_20,code=sm_20 # binary for Fermi
#CUDA_CPPFLAGS+= -gencode=arch=compute_20,code=compute_20 #Forward-compatible PTX
#
## === MPI Configuration ===
## Path to MPI installation
#MPI_BASE=/sw/analysis-x64/ompi/1.2.6/sl5_gcc4.2.0
#
## Flags for MPI compilation support.
#MPI_CPPFLAGS=-I${MPI_BASE}/include
#MPI_LIBS=-lmpi -lmpi_cxx
#
#MPI_INCLUDE?=$(MPI_BASE)/include
#MPI_LIB?=$(MPI_BASE)/lib
#MPICC?=mpicc
#MPICXX?=mpicxx
#MPI_CPPFLAGS?=-I$(MPI_INCLUDE)
#MPI_CPPFLAGS+=-I$(SHOC_ROOT)/src/mpi/common
#MPI_LDFLAGS?=-L$(MPI_LIB)
#MPI_LIBS?=-lmpi
#
## === MISC ===
## Swap modules on lens
#SWAP=$(shell module swap PrgEnv PrgEnv-gnu)
#
## g++ does not support the C99 restrict keyword, but does support
## a compiler-specific version. Uncomment if using g++
#CXXFLAGS+=-Drestrict=__restrict
#
## These flags will get you a more accurate timer
## if your OS supports clock_gettime
#CPPFLAGS+=-DUSE_CLOCK_GETTIME
#LIBS+=-lrt
#
