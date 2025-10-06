# SHOC CUDA Sort EP (Embarrassingly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# Sort EP benchmark
shoc_add_cuda_ep_executable(Sort
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../Sort.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../sort_kernel.cu
        ${CUDA_COMMON_DIR}/main.cpp
)
