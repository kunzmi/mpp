
add_subdirectory(arithmetic)

add_library(backends.cuda.image INTERFACE)
	
#target_include_directories(backends.cuda.image  PUBLIC 
#						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#						   ${PROJECT_SOURCE_DIR}/src
#						  )
#target_link_libraries(backends.cuda.image PRIVATE  
#						project_options
#						#project_warnings						
#						)
target_link_libraries(backends.cuda.image INTERFACE  
						backends.cuda.image.arithmetic
						)