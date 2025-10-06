# SHOC OpenCL Level 0 EP (Embarrassingly Parallel) MPI Benchmarks

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# BusSpeedDownload EP benchmark
shoc_add_opencl_ep_executable(BusSpeedDownload
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../BusSpeedDownload.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# BusSpeedReadback EP benchmark
shoc_add_opencl_ep_executable(BusSpeedReadback
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../BusSpeedReadback.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# DeviceMemory EP benchmark
shoc_add_opencl_ep_executable(DeviceMemory
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../DeviceMemory.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# KernelCompile EP benchmark
shoc_add_opencl_ep_executable(KernelCompile
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../KernelCompile.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# MaxFlops EP benchmark
shoc_add_opencl_ep_executable(MaxFlops
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../MaxFlops.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# QueueDelay EP benchmark
shoc_add_opencl_ep_executable(QueueDelay
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../QueueDelay.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)
