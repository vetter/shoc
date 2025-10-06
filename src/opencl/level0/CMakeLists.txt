# SHOC OpenCL Level 0 Benchmarks
# These are basic device capability tests

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# BusSpeedDownload benchmark
shoc_add_opencl_serial_executable(BusSpeedDownload
    SOURCES
        BusSpeedDownload.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# BusSpeedReadback benchmark
shoc_add_opencl_serial_executable(BusSpeedReadback
    SOURCES
        BusSpeedReadback.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# DeviceMemory benchmark
shoc_add_opencl_serial_executable(DeviceMemory
    SOURCES
        DeviceMemory.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# KernelCompile benchmark
shoc_add_opencl_serial_executable(KernelCompile
    SOURCES
        KernelCompile.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# MaxFlops benchmark
shoc_add_opencl_serial_executable(MaxFlops
    SOURCES
        MaxFlops.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# QueueDelay benchmark
shoc_add_opencl_serial_executable(QueueDelay
    SOURCES
        QueueDelay.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
