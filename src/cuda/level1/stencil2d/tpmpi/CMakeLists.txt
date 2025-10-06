# SHOC CUDA Stencil2D TP (Truly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# Stencil2D TP benchmark
shoc_add_cuda_tp_executable(Stencil2D
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../CUDAStencil.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../CommonCUDAStencilFactory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../Stencil2Dmain.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../CUDAStencilFactory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../CUDAStencilKernel.cu
        ${CMAKE_SOURCE_DIR}/src/mpi/common/MPICUDAStencil.cpp
        ${CMAKE_SOURCE_DIR}/src/mpi/common/MPICUDAStencilFactory.cpp
        ${CUDA_COMMON_DIR}/main.cpp
)
