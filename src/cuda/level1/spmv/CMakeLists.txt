# SHOC CUDA SpMV (Sparse Matrix-Vector Multiply) Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# Spmv benchmark
shoc_add_cuda_serial_executable(Spmv
    SOURCES
        Spmv.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
