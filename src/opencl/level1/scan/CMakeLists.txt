# SHOC OpenCL Scan Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernels
shoc_embed_opencl_kernel(
    scan_kernel.cl
    ${CMAKE_CURRENT_BINARY_DIR}/scan_kernel_cl.cpp
    scan_kernel
)

shoc_embed_opencl_kernel(
    addKernel.cl
    ${CMAKE_CURRENT_BINARY_DIR}/addKernel_cl.cpp
    addKernel
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# Scan benchmark
shoc_add_opencl_serial_executable(Scan
    SOURCES
        Scan.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/scan_kernel_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/addKernel_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled (both EP and TP)
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
    add_subdirectory(tpmpi)
endif()
