add_library(common 
	exception.cpp
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