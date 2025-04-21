add_library(common 
	bfloat16.cpp
	complex.cpp
	exception.cpp
	half_fp16.cpp
	scratchBufferException.cpp
	vector1.cpp
	vector2.cpp
	vector3.cpp
	vector4.cpp
	vector4A.cpp
	)

target_link_libraries(common PRIVATE  
						project_options
						project_warnings						
						)
target_include_directories(common  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
					
add_subdirectory(fileIO)	
add_subdirectory(image)
add_subdirectory(signal)