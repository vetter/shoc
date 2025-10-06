# SHOC CUDA Triad EP (Embarrassingly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# Triad EP benchmark
shoc_add_cuda_ep_executable(Triad
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../Triad.cu
        ${CUDA_COMMON_DIR}/main.cpp
)
