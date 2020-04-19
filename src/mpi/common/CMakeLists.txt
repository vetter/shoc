#
set(SHOC_COMMON_MPI_SRCS
	RandomPairs.cpp 
	MPI2DGridProgram.cpp 
	MPIHostStencil.cpp 
	MPIHostStencilFactory.cpp 
	MPIStencilUtil.cpp
	)

set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

add_library(SHOCCommonMPI STATIC ${SHOC_COMMON_MPI_SRCS})
#target_include_directories (ShOCCommon PUBLIC ${SHOC_INC_DIR})

target_include_directories(SHOCCommonMPI PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}
	)

install(TARGETS SHOCCommonMPI
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
		
# eof
