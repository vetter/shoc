# SHOC OpenCL Triad Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernel
shoc_embed_opencl_kernel(
    Triad.cl
    ${CMAKE_CURRENT_BINARY_DIR}/Triad_cl.cpp
    Triad
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# Triad benchmark
shoc_add_opencl_serial_executable(Triad
    SOURCES
        Triad.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/Triad_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
