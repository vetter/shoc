# SHOC CUDA BFS Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# BFS benchmark
shoc_add_cuda_serial_executable(BFS
    SOURCES
        BFS.cu
        bfs_kernel.cu
        Graph.cpp
        ${CUDA_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
