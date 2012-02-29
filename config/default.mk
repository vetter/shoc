# default.mk
#
# This is the default makefile configuration used
# to specify header paths and libraries needed by
# the SHOC benchmark suite.  This is the file called
# as a backup in case there is no <hostname>.mk file,
# so for a one-time machine configuration, you can
# edit this file instead.

# === OpenCL Configuration ===

# Pointer to the root of the OpenCL installation.
# If you aren't using opencl, uncomment the second line
# and skip to the next section
#OPENCL=
#OPENCL=NONE

# Flags and Libraries
# OCL_CPPFLAGS specifies flags passed to the compiler when compiling
# OpenCL code.  
# OCL_LDFLAGS specifies flags passed to the compiler when linking OpenCL
# code.
# OCL_LIBS specifies the flag to link the OpenCL library

# Always uncomment the following include directive for SHOC
#OCL_CPPFLAGS= -I$(SHOC_ROOT)/src/opencl/common

# Add custom flags here
#OCL_CPPFLAGS+= 
#OCL_LDFLAGS+=
#OCL_LIBS+=

# Examples for ATI 
#OCL_CPPFLAGS+=-I${OPENCL}/include
#OCL_LDFLAGS=-Wl,-rpath=${OPENCL}/lib/x86_64
#OCL_LIBS=-L${OPENCL}/lib/x86_64 -lOpenCL

# Examples for NV
#OCL_CPPFLAGS+=-I$(OPENCL)/OpenCL/common/inc -I$(OPENCL)/shared/inc
#OCL_LDFLAGS=
#OCL_LIBS=-lOpenCL

# Examples for Snow Leopard
#OCL_CPPFLAGS=-I$(OPENCL)/Headers
#OCL_LDFLAGS=-Wl -framework OpenCL
#OCL_LIBS=

# === CUDA Configuration ===

# Path to the root of the CUDA installation.
# If you aren't using CUDA, uncomment the second line
# and skip to the next section
#CUDA=/usr/local/cuda
#CUDA=NONE

# CUDA Basics, you typically don't have to change these
#NVCC?=$(CUDA)/bin/nvcc
#CUDA_CXX?=$(NVCC)
#CUDA_LD?=$(NVCC)
#CUDA_INC?=-I$(CUDA)/include

# Set the rpath to the CUDA lib folder (where libcuda and libcudart are)
#CUDA_RPATH=${CUDA}/lib
#CUDA_RPATH=${CUDA}/lib64
#CUDA_LDFLAGS=--linker-options -rpath=$(CUDA_RPATH)

# Compilation flags for CUDA
# Always uncomment the next line to specify include paths
#CUDA_CPPFLAGS=-I${CUDA}/include -I${SHOC_ROOT}/src/cuda/include

# Go ahead and generate all necessary CUDA binaries/ptx
# If you are unsure of which version you need, uncomment all the lines below
#CUDA_CPPFLAGS+= -gencode=arch=compute_10,code=sm_10 # binary for compute 1
#CUDA_CPPFLAGS+= -gencode=arch=compute_11,code=sm_11 # binary for compute 1.1
#CUDA_CPPFLAGS+= -gencode=arch=compute_13,code=sm_13 # binary for Tesla
#CUDA_CPPFLAGS+= -gencode=arch=compute_20,code=sm_20 # binary for Fermi
#CUDA_CPPFLAGS+= -gencode=arch=compute_20,code=compute_20 #Forward-compatible PTX

# === MPI Configuration ===
# Path to MPI installation
# If you aren't using MPI, uncomment the second line
# and skip to the next section
#MPI_BASE=
#MPI_BASE=NONE

# Flags for MPI compilation support.
# Path to mpi.h, typically /include or /include64
#MPI_CPPFLAGS=-I${MPI_BASE}/include

# Always include the SHOC mpi include directory
#MPI_CPPFLAGS+=-I$(SHOC_ROOT)/src/mpi/common

# Path to MPI libarries, typically /lib or /lib64
#MPI_LIB=$(MPI_BASE)/lib

# Flags to include MPI library (typically -lmpi or "-lmpi -lmpi_cxx")
#MPI_LIBS=-lmpi

# MPI Basics, normally these don't need to be changed
#MPICC?=mpicc
#MPICXX?=mpicxx
#MPI_LDFLAGS?=-L$(MPI_LIB)

# === MISC ===

# Uncomment this to use gettimeofday() (recommended)
#CPPFLAGS+=-DUSE_GETTIMEOFDAY

# Uncomment these to 2 use clock_gettime() if your OS supports it
#CPPFLAGS+=-DUSE_CLOCK_GETTIME
#LIBS+=-lrt
