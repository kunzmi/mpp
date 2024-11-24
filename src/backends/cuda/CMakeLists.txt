add_library(backends.cuda 
	cudaException.cpp
	event.cpp
	stream.cpp
	streamCtx.cpp
	)
	
target_include_directories(backends.cuda  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.cuda PRIVATE  
						project_options
						project_warnings						
						)
target_link_libraries(backends.cuda PUBLIC  
						common
						)

add_subdirectory(image)