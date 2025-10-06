# SHOC CUDA Reduction EP (Embarrassingly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# Reduction EP benchmark
shoc_add_cuda_ep_executable(Reduction
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../Reduction.cu
        ${CUDA_COMMON_DIR}/main.cpp
)
