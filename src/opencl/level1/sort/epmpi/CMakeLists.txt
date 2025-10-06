# SHOC OpenCL Sort EP (Embarrassingly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernel (same as parent)
shoc_embed_opencl_kernel(
    ${CMAKE_CURRENT_SOURCE_DIR}/../sort.cl
    ${CMAKE_CURRENT_BINARY_DIR}/sort_cl.cpp
    sort
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# Sort EP benchmark
shoc_add_opencl_ep_executable(Sort
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../Sort.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/sort_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)
