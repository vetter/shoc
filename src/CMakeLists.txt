
add_subdirectory(common)

if(${ENABLE_MPI})
	add_subdirectory(mpi)
endif()

if(${ENABLE_OPENCL})
	add_subdirectory(opencl)
endif()

