# SHOC CUDA Level 0 EP (Embarrassingly Parallel) MPI Benchmarks
# These are the same benchmarks as Serial but run with MPI

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# Each EP benchmark uses the same .cu file from parent directory
# but links with MPI and gets installed to bin/EP/CUDA

# BusSpeedDownload EP benchmark
shoc_add_cuda_ep_executable(BusSpeedDownload
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../BusSpeedDownload.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# BusSpeedReadback EP benchmark
shoc_add_cuda_ep_executable(BusSpeedReadback
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../BusSpeedReadback.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# DeviceMemory EP benchmark
shoc_add_cuda_ep_executable(DeviceMemory
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../DeviceMemory.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# MaxFlops EP benchmark
shoc_add_cuda_ep_executable(MaxFlops
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../MaxFlops.cu
        ${CUDA_COMMON_DIR}/main.cpp
)
