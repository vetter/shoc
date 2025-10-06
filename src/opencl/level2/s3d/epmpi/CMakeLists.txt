# SHOC OpenCL S3D EP (Embarrassingly Parallel) MPI Benchmark

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Embed the OpenCL kernels (same as parent)
shoc_embed_opencl_kernel(
    ${CMAKE_CURRENT_SOURCE_DIR}/../ratt.cl
    ${CMAKE_CURRENT_BINARY_DIR}/ratt_cl.cpp
    ratt
)

shoc_embed_opencl_kernel(
    ${CMAKE_CURRENT_SOURCE_DIR}/../ratt2.cl
    ${CMAKE_CURRENT_BINARY_DIR}/ratt2_cl.cpp
    ratt2
)

shoc_embed_opencl_kernel(
    ${CMAKE_CURRENT_SOURCE_DIR}/../ratx.cl
    ${CMAKE_CURRENT_BINARY_DIR}/ratx_cl.cpp
    ratx
)

shoc_embed_opencl_kernel(
    ${CMAKE_CURRENT_SOURCE_DIR}/../ratx2.cl
    ${CMAKE_CURRENT_BINARY_DIR}/ratx2_cl.cpp
    ratx2
)

shoc_embed_opencl_kernel(
    ${CMAKE_CURRENT_SOURCE_DIR}/../rdwdot.cl
    ${CMAKE_CURRENT_BINARY_DIR}/rdwdot_cl.cpp
    rdwdot
)

shoc_embed_opencl_kernel(
    ${CMAKE_CURRENT_SOURCE_DIR}/../gr_base.cl
    ${CMAKE_CURRENT_BINARY_DIR}/gr_base_cl.cpp
    gr_base
)

set(OPENCL_COMMON_DIR ${CMAKE_SOURCE_DIR}/src/opencl/common)

# S3D EP benchmark
shoc_add_opencl_ep_executable(S3D
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../S3D.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ratt_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ratt2_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ratx_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ratx2_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/rdwdot_cl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/gr_base_cl.cpp
        ${OPENCL_COMMON_DIR}/main.cpp
)
