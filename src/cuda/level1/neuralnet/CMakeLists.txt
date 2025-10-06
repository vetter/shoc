# SHOC CUDA NeuralNet Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

set(CUDA_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/cuda/common)

# NeuralNet benchmark
shoc_add_cuda_serial_executable(NeuralNet
    SOURCES
        NeuralNet.cu
        ${CUDA_COMMON_DIR}/main.cpp
)

# Install data files if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nn_data.zip")
    install(CODE "
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/nn_data.zip
            WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin
        )
    ")
endif()

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
