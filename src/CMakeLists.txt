# SHOC source directory

# Always build common library
add_subdirectory(common)

# Build CUDA components if enabled
if(SHOC_CUDA_ENABLED)
    add_subdirectory(cuda)
endif()

# Build OpenCL components if enabled
if(SHOC_OPENCL_ENABLED)
    add_subdirectory(opencl)
endif()

# Build MPI components if enabled
if(SHOC_MPI_ENABLED)
    add_subdirectory(mpi)
endif()

# Build stability tests if enabled
if(SHOC_BUILD_STABILITY)
    add_subdirectory(stability)
endif()
