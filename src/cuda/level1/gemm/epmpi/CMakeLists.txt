# SHOC CUDA GEMM EP (Embarrassingly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# GEMM EP benchmark
shoc_add_cuda_ep_executable(GEMM
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../GEMM.cpp
        ${CUDA_COMMON_DIR}/main.cpp
)
