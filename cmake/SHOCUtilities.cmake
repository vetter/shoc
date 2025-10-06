# SHOC CMake utility functions

# Function to embed OpenCL kernel source into C++ code
# This creates a custom command that generates a .cpp file from a .cl file
function(shoc_embed_opencl_kernel INPUT_CL OUTPUT_CPP KERNEL_NAME)
    get_filename_component(INPUT_CL_ABS "${INPUT_CL}" ABSOLUTE)
    get_filename_component(OUTPUT_CPP_ABS "${OUTPUT_CPP}" ABSOLUTE)

    add_custom_command(
        OUTPUT "${OUTPUT_CPP_ABS}"
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE="${INPUT_CL_ABS}"
            -DOUTPUT_FILE="${OUTPUT_CPP_ABS}"
            -DKERNEL_NAME="${KERNEL_NAME}"
            -P "${CMAKE_SOURCE_DIR}/cmake/EmbedOpenCLKernel.cmake"
        DEPENDS "${INPUT_CL_ABS}"
        COMMENT "Embedding OpenCL kernel ${KERNEL_NAME} from ${INPUT_CL}"
        VERBATIM
    )
endfunction()

# Function to add a CUDA serial benchmark executable
# Creates an executable and installs it to bin/Serial/CUDA
function(shoc_add_cuda_serial_executable TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPENDS" ${ARGN})

    add_executable(${TARGET} ${ARG_SOURCES})

    target_link_libraries(${TARGET} PRIVATE
        SHOCCommon
        CUDA::cudart
        CUDA::cublas
        CUDA::cufft
        ${ARG_DEPENDS}
    )

    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/cuda/common
    )

    install(TARGETS ${TARGET}
        RUNTIME DESTINATION bin/Serial/CUDA
    )
endfunction()

# Function to add an OpenCL serial benchmark executable
# Creates an executable and installs it to bin/Serial/OpenCL
function(shoc_add_opencl_serial_executable TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPENDS" ${ARGN})

    add_executable(${TARGET} ${ARG_SOURCES})

    target_link_libraries(${TARGET} PRIVATE
        SHOCCommon
        SHOCCommonOpenCL
        OpenCL::OpenCL
        ${ARG_DEPENDS}
    )

    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/opencl/common
        ${CMAKE_SOURCE_DIR}/src/common
    )

    install(TARGETS ${TARGET}
        RUNTIME DESTINATION bin/Serial/OpenCL
    )
endfunction()

# Function to add a CUDA EP (Embarrassingly Parallel) MPI benchmark executable
# Creates an executable and installs it to bin/EP/CUDA
function(shoc_add_cuda_ep_executable TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPENDS" ${ARGN})

    add_executable(${TARGET} ${ARG_SOURCES})

    target_link_libraries(${TARGET} PRIVATE
        SHOCCommon
        MPI::MPI_CXX
        CUDA::cudart
        CUDA::cublas
        CUDA::cufft
        ${ARG_DEPENDS}
    )

    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/cuda/common
        ${CMAKE_SOURCE_DIR}/src/mpi/common
    )

    target_compile_definitions(${TARGET} PRIVATE PARALLEL)

    install(TARGETS ${TARGET}
        RUNTIME DESTINATION bin/EP/CUDA
    )
endfunction()

# Function to add an OpenCL EP (Embarrassingly Parallel) MPI benchmark executable
# Creates an executable and installs it to bin/EP/OpenCL
function(shoc_add_opencl_ep_executable TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPENDS" ${ARGN})

    add_executable(${TARGET} ${ARG_SOURCES})

    target_link_libraries(${TARGET} PRIVATE
        SHOCCommon
        SHOCCommonOpenCL
        MPI::MPI_CXX
        OpenCL::OpenCL
        ${ARG_DEPENDS}
    )

    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/opencl/common
        ${CMAKE_SOURCE_DIR}/src/mpi/common
        ${CMAKE_SOURCE_DIR}/src/common
    )

    target_compile_definitions(${TARGET} PRIVATE PARALLEL)

    install(TARGETS ${TARGET}
        RUNTIME DESTINATION bin/EP/OpenCL
    )
endfunction()

# Function to add a CUDA TP (Truly Parallel) MPI benchmark executable
# Creates an executable and installs it to bin/TP/CUDA
function(shoc_add_cuda_tp_executable TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPENDS" ${ARGN})

    add_executable(${TARGET} ${ARG_SOURCES})

    target_link_libraries(${TARGET} PRIVATE
        SHOCCommon
        MPI::MPI_CXX
        CUDA::cudart
        CUDA::cublas
        CUDA::cufft
        ${ARG_DEPENDS}
    )

    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/cuda/common
        ${CMAKE_SOURCE_DIR}/src/mpi/common
    )

    target_compile_definitions(${TARGET} PRIVATE PARALLEL)

    install(TARGETS ${TARGET}
        RUNTIME DESTINATION bin/TP/CUDA
    )
endfunction()

# Function to add an OpenCL TP (Truly Parallel) MPI benchmark executable
# Creates an executable and installs it to bin/TP/OpenCL
function(shoc_add_opencl_tp_executable TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPENDS" ${ARGN})

    add_executable(${TARGET} ${ARG_SOURCES})

    target_link_libraries(${TARGET} PRIVATE
        SHOCCommon
        SHOCCommonOpenCL
        MPI::MPI_CXX
        OpenCL::OpenCL
        ${ARG_DEPENDS}
    )

    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/opencl/common
        ${CMAKE_SOURCE_DIR}/src/mpi/common
        ${CMAKE_SOURCE_DIR}/src/common
    )

    target_compile_definitions(${TARGET} PRIVATE PARALLEL)

    install(TARGETS ${TARGET}
        RUNTIME DESTINATION bin/TP/OpenCL
    )
endfunction()
