# SHOC CUDA Scan TP (Truly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# Scan TP benchmark
shoc_add_cuda_tp_executable(Scan
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../Scan.cu
        ${CUDA_COMMON_DIR}/main.cpp
)
