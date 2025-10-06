# SHOC OpenCL Stencil2D Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernel
shoc_embed_opencl_kernel(
    Stencil2D.cl
    ${CMAKE_CURRENT_BINARY_DIR}/Stencil2D_cl.cpp
    Stencil2D
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# Stencil2D benchmark
shoc_add_opencl_serial_executable(Stencil2D
    SOURCES
        OpenCLStencil.cpp
        CommonOpenCLStencilFactory.cpp
        Stencil2Dmain.cpp
        OpenCLStencilFactory.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/Stencil2D_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled (only TP for Stencil2D)
if(SHOC_MPI_ENABLED)
    add_subdirectory(tpmpi)
endif()
