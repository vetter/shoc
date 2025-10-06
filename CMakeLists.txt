cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# SHOC - Scalable HeterOgeneous Computing Benchmark Suite
project(SHOC VERSION 1.1.5 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default installation prefix to build directory (matching autotools behavior)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "Install prefix" FORCE)
endif()

# Build options
option(SHOC_BUILD_CUDA "Build CUDA benchmarks" ON)
option(SHOC_BUILD_OPENCL "Build OpenCL benchmarks" ON)
option(SHOC_BUILD_MPI "Build MPI parallel benchmarks" ON)
option(SHOC_BUILD_STABILITY "Build stability tests (requires CUDA)" ON)
option(SHOC_BUILD_DOC "Build documentation (requires pdflatex, bibtex, latexmk)" OFF)

# Add cmake modules directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Include SHOC utility functions
include(SHOCUtilities)

# Platform-specific configuration
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # On Linux, clock_gettime requires rt library
    find_library(RT_LIBRARY rt)
    if(RT_LIBRARY)
        set(PLATFORM_LIBS ${RT_LIBRARY})
    endif()
endif()

# Check for required headers and functions
include(CheckIncludeFile)
include(CheckFunctionExists)
check_include_file("sys/time.h" HAVE_SYS_TIME_H)
check_include_file("sys/timeb.h" HAVE_SYS_TIMEB_H)
check_function_exists(clock_gettime HAVE_CLOCK_GETTIME)
check_function_exists(gettimeofday HAVE_GETTIMEOFDAY)

# CUDA support
set(SHOC_CUDA_ENABLED FALSE)
if(SHOC_BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(SHOC_CUDA_ENABLED TRUE)

        # Set CUDA standard
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        # CUDA architecture configuration
        # Allow user to specify architectures, otherwise use "native" or common defaults
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # Try to detect GPU architecture, otherwise use common values
            set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80" CACHE STRING
                "CUDA architectures to compile for (semicolon-separated list, e.g., '50;60;70' or 'native')")
        endif()

        # Find CUDA toolkit components
        find_package(CUDAToolkit REQUIRED)

        message(STATUS "CUDA enabled - Version: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(WARNING "CUDA compiler not found, disabling CUDA support")
        set(SHOC_BUILD_CUDA OFF)
    endif()
endif()

# OpenCL support
set(SHOC_OPENCL_ENABLED FALSE)
if(SHOC_BUILD_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        set(SHOC_OPENCL_ENABLED TRUE)
        message(STATUS "OpenCL enabled - Version: ${OpenCL_VERSION_STRING}")
    else()
        message(WARNING "OpenCL not found, disabling OpenCL support")
        set(SHOC_BUILD_OPENCL OFF)
    endif()
endif()

# MPI support
set(SHOC_MPI_ENABLED FALSE)
if(SHOC_BUILD_MPI)
    # MPI only makes sense if we have CUDA or OpenCL
    if(SHOC_CUDA_ENABLED OR SHOC_OPENCL_ENABLED)
        find_package(MPI COMPONENTS CXX)
        if(MPI_CXX_FOUND)
            set(SHOC_MPI_ENABLED TRUE)
            message(STATUS "MPI enabled - Version: ${MPI_CXX_VERSION}")
        else()
            message(WARNING "MPI not found, disabling MPI support")
            set(SHOC_BUILD_MPI OFF)
        endif()
    else()
        message(STATUS "MPI support requires CUDA or OpenCL to be enabled, skipping MPI")
        set(SHOC_BUILD_MPI OFF)
    endif()
endif()

# Stability tests
if(SHOC_BUILD_STABILITY AND NOT SHOC_CUDA_ENABLED)
    message(WARNING "Stability tests require CUDA, disabling stability tests")
    set(SHOC_BUILD_STABILITY OFF)
endif()

# Summary
message(STATUS "=== SHOC Configuration Summary ===")
message(STATUS "CUDA support:      ${SHOC_CUDA_ENABLED}")
message(STATUS "OpenCL support:    ${SHOC_OPENCL_ENABLED}")
message(STATUS "MPI support:       ${SHOC_MPI_ENABLED}")
message(STATUS "Stability tests:   ${SHOC_BUILD_STABILITY}")
message(STATUS "Documentation:     ${SHOC_BUILD_DOC}")
message(STATUS "Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "==================================")

# Check that at least one compute framework is enabled
if(NOT SHOC_CUDA_ENABLED AND NOT SHOC_OPENCL_ENABLED)
    message(FATAL_ERROR "At least one of CUDA or OpenCL must be enabled to build SHOC")
endif()

# Generate config header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config/config.h"
)

# Add subdirectories
add_subdirectory(src)

if(SHOC_BUILD_DOC)
    add_subdirectory(doc)
endif()

add_subdirectory(tools)

# Installation of shocdriver script
install(PROGRAMS tools/shocdriver
    DESTINATION bin
)
