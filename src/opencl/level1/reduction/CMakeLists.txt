# SHOC OpenCL Reduction Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernel
shoc_embed_opencl_kernel(
    reduction.cl
    ${CMAKE_CURRENT_BINARY_DIR}/reduction_cl.cpp
    reduction
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# Reduction benchmark
shoc_add_opencl_serial_executable(Reduction
    SOURCES
        Reduction.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/reduction_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled (both EP and TP)
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
    add_subdirectory(tpmpi)
endif()
