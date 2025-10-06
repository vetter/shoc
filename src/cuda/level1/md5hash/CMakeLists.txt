# SHOC CUDA MD5Hash Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# MD5Hash benchmark
shoc_add_cuda_serial_executable(MD5Hash
    SOURCES
        MD5Hash.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
