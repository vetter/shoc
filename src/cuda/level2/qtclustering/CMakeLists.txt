# SHOC CUDA QTClustering Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# QTC (Quality Threshold Clustering) benchmark
shoc_add_cuda_serial_executable(QTC
    SOURCES
        QTC.cu
        comm.cpp
        libdata.cpp
        ${CUDA_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled (only TP for QTC)
if(SHOC_MPI_ENABLED)
    add_subdirectory(tpmpi)
endif()
