# SHOC OpenCL FFT Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernel
shoc_embed_opencl_kernel(
    fft.cl
    ${CMAKE_CURRENT_BINARY_DIR}/fft_cl.cpp
    fft
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# FFT benchmark
shoc_add_opencl_serial_executable(FFT
    SOURCES
        FFT.cpp
        fftlib.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/fft_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
