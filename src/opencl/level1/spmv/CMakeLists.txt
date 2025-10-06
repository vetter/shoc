# SHOC OpenCL SpMV (Sparse Matrix-Vector Multiply) Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernel
shoc_embed_opencl_kernel(
    spmv.cl
    ${CMAKE_CURRENT_BINARY_DIR}/spmv_cl.cpp
    spmv
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# Spmv benchmark
shoc_add_opencl_serial_executable(Spmv
    SOURCES
        Spmv.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/spmv_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
