# SHOC CUDA FFT EP (Embarrassingly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# FFT EP benchmark
shoc_add_cuda_ep_executable(FFT
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../FFT.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../fftlib.cu
        ${CUDA_COMMON_DIR}/main.cpp
)
