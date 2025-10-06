# SHOC CUDA BFS EP (Embarrassingly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# BFS EP benchmark
shoc_add_cuda_ep_executable(BFS
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../BFS.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../bfs_kernel.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../Graph.cpp
        ${CUDA_COMMON_DIR}/main.cpp
)
