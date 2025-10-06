# SHOC CUDA Reduction Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# Reduction benchmark
shoc_add_cuda_serial_executable(Reduction
    SOURCES
        Reduction.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled (both EP and TP)
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
    add_subdirectory(tpmpi)
endif()
