# SHOC OpenCL S3D Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernels
shoc_embed_opencl_kernel(
    ratt.cl
    ${CMAKE_CURRENT_BINARY_DIR}/ratt_cl.cpp
    ratt
)

shoc_embed_opencl_kernel(
    ratt2.cl
    ${CMAKE_CURRENT_BINARY_DIR}/ratt2_cl.cpp
    ratt2
)

shoc_embed_opencl_kernel(
    ratx.cl
    ${CMAKE_CURRENT_BINARY_DIR}/ratx_cl.cpp
    ratx
)

shoc_embed_opencl_kernel(
    ratx2.cl
    ${CMAKE_CURRENT_BINARY_DIR}/ratx2_cl.cpp
    ratx2
)

shoc_embed_opencl_kernel(
    rdwdot.cl
    ${CMAKE_CURRENT_BINARY_DIR}/rdwdot_cl.cpp
    rdwdot
)

shoc_embed_opencl_kernel(
    gr_base.cl
    ${CMAKE_CURRENT_BINARY_DIR}/gr_base_cl.cpp
    gr_base
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# S3D benchmark
shoc_add_opencl_serial_executable(S3D
    SOURCES
        S3D.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ratt_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ratt2_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ratx_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ratx2_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/rdwdot_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/gr_base_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)

# Add MPI versions if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
