# SHOC CUDA S3D EP (Embarrassingly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# S3D EP benchmark
shoc_add_cuda_ep_executable(S3D
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../S3D.cu
        ${CUDA_COMMON_DIR}/main.cpp
)
