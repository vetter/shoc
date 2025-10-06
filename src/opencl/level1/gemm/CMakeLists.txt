# SHOC OpenCL GEMM Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# GEMM benchmark (no kernel embedding needed - uses vendor BLAS)
shoc_add_opencl_serial_executable(GEMM
    SOURCES
        GEMM.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
