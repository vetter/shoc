# SHOC OpenCL Common Library
# This library contains OpenCL-specific utilities shared across OpenCL benchmarks

add_library(SHOCCommonOpenCL STATIC
    OpenCLDeviceInfo.cpp
    OpenCLPlatform.cpp
    OpenCLNodePlatformContainer.cpp
    Event.cpp
)

target_include_directories(SHOCCommonOpenCL PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/common>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/config>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(SHOCCommonOpenCL PUBLIC
    SHOCCommon
    OpenCL::OpenCL
)

# Installation
install(TARGETS SHOCCommonOpenCL
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
