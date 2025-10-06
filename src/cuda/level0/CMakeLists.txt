# SHOC CUDA Level 0 Benchmarks
# These are basic device capability tests

# Include the SHOC utilities
include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Each benchmark uses main.cpp from cuda/common plus its own .cu file
set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# BusSpeedDownload benchmark
shoc_add_cuda_serial_executable(BusSpeedDownload
    SOURCES
        BusSpeedDownload.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# BusSpeedReadback benchmark
shoc_add_cuda_serial_executable(BusSpeedReadback
    SOURCES
        BusSpeedReadback.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# DeviceMemory benchmark
shoc_add_cuda_serial_executable(DeviceMemory
    SOURCES
        DeviceMemory.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# MaxFlops benchmark
shoc_add_cuda_serial_executable(MaxFlops
    SOURCES
        MaxFlops.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
