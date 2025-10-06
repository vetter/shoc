# SHOC OpenCL Level 0 Benchmarks
# These are basic device capability tests

include(${CMAKE_SOURCE_DIR}/cmake/SHOCUtilities.cmake)

# Placeholder - will be populated with OpenCL level0 benchmarks
# Similar structure to CUDA level0

if(SHOC_MPI_ENABLED)
    add_subdirectory(epmpi)
endif()
