# CMake script to embed OpenCL kernel source files as C++ string literals
# This replicates the functionality of the autotools build system's sed commands
#
# Usage: cmake -P EmbedOpenCLKernel.cmake <input.cl> <output_cl.cpp> <kernel_name>

if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE OR NOT DEFINED KERNEL_NAME)
    message(FATAL_ERROR "Usage: cmake -DINPUT_FILE=<file.cl> -DOUTPUT_FILE=<file_cl.cpp> -DKERNEL_NAME=<name> -P EmbedOpenCLKernel.cmake")
endif()

if(NOT EXISTS "${INPUT_FILE}")
    message(FATAL_ERROR "Input file does not exist: ${INPUT_FILE}")
endif()

# Read the OpenCL kernel source file
file(READ "${INPUT_FILE}" KERNEL_SOURCE)

# Remove carriage returns (Windows line endings)
string(REPLACE "\r" "" KERNEL_SOURCE "${KERNEL_SOURCE}")

# Escape backslashes
string(REPLACE "\\" "\\\\" KERNEL_SOURCE "${KERNEL_SOURCE}")

# Escape double quotes
string(REPLACE "\"" "\\\"" KERNEL_SOURCE "${KERNEL_SOURCE}")

# Split into lines and add quotes and newlines
string(REPLACE "\n" "\\n\"\n\"" KERNEL_SOURCE "${KERNEL_SOURCE}")

# Write the output C++ file
file(WRITE "${OUTPUT_FILE}" "const char *cl_source_${KERNEL_NAME} =\n\"${KERNEL_SOURCE}\";\n")

message(STATUS "Embedded OpenCL kernel: ${INPUT_FILE} -> ${OUTPUT_FILE}")
