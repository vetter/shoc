# SHOC CUDA Stencil2D Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# Stencil2D benchmark
shoc_add_cuda_serial_executable(Stencil2D
    SOURCES
        CUDAStencil.cpp
        CommonCUDAStencilFactory.cpp
        Stencil2Dmain.cpp
        CUDAStencilFactory.cpp
        CUDAStencilKernel.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled (only TP for Stencil2D)
if(SHOC_MPI_ENABLED)
    add_subdirectory(tpmpi)
endif()
