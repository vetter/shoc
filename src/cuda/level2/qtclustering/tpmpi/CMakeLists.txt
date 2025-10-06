# SHOC CUDA QTClustering TP (Truly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# QTC TP benchmark
shoc_add_cuda_tp_executable(QTC
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../QTC.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../comm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../libdata.cpp
        ${CMAKE_SOURCE_DIR}/src/mpi/common/OptionParser.cpp
        ${CUDA_COMMON_DIR}/main.cpp
)
