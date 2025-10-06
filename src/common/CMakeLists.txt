# SHOC Common Library
# This library contains utilities shared across all benchmarks

add_library(SHOCCommon STATIC
    CTimer.cpp
    ResultDatabase.cpp
    OptionParser.cpp
    Option.cpp
    Timer.cpp
    ProgressBar.cpp
    InvalidArgValue.cpp
    Matrix2DStatics.cpp
)

target_include_directories(SHOCCommon PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/config>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(SHOCCommon PUBLIC
    ${PLATFORM_LIBS}
)

# Installation
install(TARGETS SHOCCommon
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
