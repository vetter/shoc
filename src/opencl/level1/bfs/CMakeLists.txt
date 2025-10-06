# SHOC OpenCL BFS Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernels
shoc_embed_opencl_kernel(
    bfs_uiuc_spill.cl
    ${CMAKE_CURRENT_BINARY_DIR}/bfs_uiuc_spill_cl.cpp
    bfs_uiuc_spill
)

shoc_embed_opencl_kernel(
    bfs_iiit.cl
    ${CMAKE_CURRENT_BINARY_DIR}/bfs_iiit_cl.cpp
    bfs_iiit
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# BFS benchmark
shoc_add_opencl_serial_executable(BFS
    SOURCES
        BFS.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/bfs_uiuc_spill_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/bfs_iiit_cl.cpp
        Graph.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
