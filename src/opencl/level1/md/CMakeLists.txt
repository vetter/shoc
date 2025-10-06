# SHOC OpenCL MD (Molecular Dynamics) Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernel
shoc_embed_opencl_kernel(
    md.cl
    ${CMAKE_CURRENT_BINARY_DIR}/md_cl.cpp
    md
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# MD benchmark
shoc_add_opencl_serial_executable(MD
    SOURCES
        MD.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/md_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
