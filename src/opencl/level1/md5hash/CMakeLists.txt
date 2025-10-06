# SHOC OpenCL MD5Hash Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernel
shoc_embed_opencl_kernel(
    md5hash.cl
    ${CMAKE_CURRENT_BINARY_DIR}/md5hash_cl.cpp
    md5hash
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# MD5Hash benchmark
shoc_add_opencl_serial_executable(MD5Hash
    SOURCES
        MD5Hash.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/md5hash_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
