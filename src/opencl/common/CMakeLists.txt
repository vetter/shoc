#
set(SHOC_COMMON_OpenCL_SRCS
	OpenCLDeviceInfo.cpp 
	OpenCLPlatform.cpp 
	OpenCLNodePlatformContainer.cpp 
	Event.cpp
	)

#set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

add_library(libSHOCCommonOpenCL STATIC ${SHOC_COMMON_OpenCL_SRCS})
target_include_directories(libSHOCCommonOpenCL PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}
	${OpenCL_INCLUDE_DIRS}
	)

install(TARGETS libSHOCCommonOpenCL
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
		
# eof
