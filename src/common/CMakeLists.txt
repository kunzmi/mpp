
add_subdirectory(image)

add_library(common SHARED
	bfloat16.cpp
	complex.cpp
	errorMessageSingleton.cpp
	exception.cpp
	half_fp16.cpp
	scratchBufferException.cpp
	vector1.cpp
	vector2.cpp
	vector3.cpp
	vector4.cpp
	vector4A.cpp
	)
target_compile_definitions(common PRIVATE -DMPP_COMMON -DENABLE_CUDA_BFLOAT -DENABLE_CUDA_HALFFLOAT)  

target_link_libraries(common PRIVATE  
						project_options
						project_warnings						
						)
target_link_libraries(common PUBLIC  
						common.image						
						)

target_include_directories(common  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )

set_target_properties(common PROPERTIES OUTPUT_NAME mpp-common)
					
add_subdirectory(fileIO)	