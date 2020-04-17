
set(SHOC_COMMON_SRCS
	CTimer.cpp 
    ResultDatabase.cpp 
    OptionParser.cpp 
    Option.cpp 
    Timer.cpp 
    ProgressBar.cpp 
    InvalidArgValue.cpp
    Matrix2DStatics.cpp
	)

add_library(ShOCCommon STATIC ${SHOC_COMMON_SRCS})
target_include_directories (ShOCCommon PUBLIC ${SHOC_INC_DIR})

#target_include_directories(ShOCCommon $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/shoc>)


