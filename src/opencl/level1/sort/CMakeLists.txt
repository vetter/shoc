# SHOC OpenCL Sort Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernel
shoc_embed_opencl_kernel(
    sort.cl
    ${CMAKE_CURRENT_BINARY_DIR}/sort_cl.cpp
    sort
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# Sort benchmark
shoc_add_opencl_serial_executable(Sort
    SOURCES
        Sort.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/sort_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
