cmake_minimum_required(VERSION 3.8)

include(CMakePrintHelpers)

project(SHOC 
	VERSION 3.0.1.202004
	DESCRIPTION "Scalable Heterogeneous Computing Benchmark Suite"
	LANGUAGES CXX)

option(ENABLE_CUDA    "Cuda"    OFF)
option(ENABLE_OPENMP  "OpenMP"  OFF)
option(ENABLE_OPENACC "OpenACC" OFF)
option(ENABLE_OPENCL  "OpenCL"  OFF)
option(ENABLE_SYCL    "SYCL"    OFF)

set(SHOC_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(SHOC_INC_DIR ${PROJECT_BINARY_DIR}/include/shoc)
#cmake_print_variables(PROJECT_SOURCE_DIR)

include_directories(${SHOC_INC_DIR})

add_subdirectory(src)
add_subdirectory(docs)
add_subdirectory(scripts)
add_subdirectory(tests)
add_subdirectory(tools)

# add_library(calclib STATIC src/calclib.cpp include/calc/lib.hpp)
# target_include_directories(calclib PUBLIC include)
# target_compile_features(calclib PUBLIC cxx_std_11)

# list(APPEND MY_SRC apps/calc.cpp)

# add_executable(calc ${MY_SRC})
# target_link_libraries(calc PUBLIC calclib)

configure_file (
    "${PROJECT_SOURCE_DIR}/include/shoc/shoci.h.in"
    "${SHOC_INC_DIR}/shoci.h"
)
