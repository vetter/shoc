cmake_minimum_required(VERSION 3.8)
project(SHOC
		VERSION 3.0.1.202004
		DESCRIPTION "Scalable Heterogeneous Computing Benchmark Suite"
		LANGUAGES CXX)

include(CMakePrintHelpers)
set(cmake_verbose 1)          # set to 1 if you need to debug the cmake build

########################
#
# Set up some global build variables
#
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

set(SHOC_SRC_DIR 	 			${PROJECT_SOURCE_DIR}/src)
set(SHOC_INC_DIR 	 			${PROJECT_SOURCE_DIR}/include/shoc)
set(SHOC_INSTALL_DIR 		${PROJECT_BINARY_DIR})
set(CMAKE_INSTALL_PREFIX ${SHOC_INSTALL_DIR})

include_directories(${SHOC_INC_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src/common)

if(${cmake_verbose})
	cmake_print_variables(PROJECT_SOURCE_DIR)
	cmake_print_variables(PROJECT_BINARY_DIR)
	cmake_print_variables(PROJECT_INSTALL_DIR)
	cmake_print_variables(SHOC_INC_DIR)
	cmake_print_variables(CMAKE_BUILD_TYPE)
endif()

########################
#
# Select programming models.
#
# enable distributed execution with MPI (orthogonal to node programming model selection)
#
option(ENABLE_MPI     	"MPI"    	ON )

# enable node programming model
option(ENABLE_CUDA    	"CUDA"    	OFF)
option(ENABLE_OPENACC 	"OpenACC" 	OFF)
option(ENABLE_OPENCL  	"OpenCL"  	OFF)
option(ENABLE_OPENMP  	"OpenMP"  	OFF )
option(ENABLE_triSYCL   "triSYCL"	OFF)

if(${cmake_verbose})
	cmake_print_variables(ENABLE_CUDA   )
	cmake_print_variables(ENABLE_OPENACC)
	cmake_print_variables(ENABLE_OPENCL )
	cmake_print_variables(ENABLE_OPENMP )
	cmake_print_variables(ENABLE_MPI )
	cmake_print_variables(ENABLE_triSYCL   )
endif()

if(${ENABLE_CUDA})
	find_package(CUDAToolkit REQUIRED)
endif()
if(${ENABLE_OPENACC})
	find_package(OpenACC REQUIRED)
endif()
if(${ENABLE_OPENCL})
	find_package(OpenCL REQUIRED)
endif()
if(${ENABLE_OPENMP})
	find_package(OpenMP REQUIRED)
endif()
if(${ENABLE_MPI})
	find_package(MPI REQUIRED)
endif()
if(${ENABLE_triSYCL})
	find_package(TriSYCL REQUIRED)
endif()

########################
#
# Add our subdirs
#
add_subdirectory(src)
add_subdirectory(docs)
add_subdirectory(scripts)
add_subdirectory(tests)
add_subdirectory(tools)

########################
#
# Create our internal header file based on our config.
#
configure_file (
    "${PROJECT_SOURCE_DIR}/include/shoc/shoci.h.in"
    "${SHOC_INC_DIR}/shoci.h"
)

# eof
