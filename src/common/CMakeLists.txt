add_library(common 
	exception.cpp
	)

target_link_libraries(common PRIVATE  
						project_options
						project_warnings						
						)
					
add_subdirectory(fileIO)	
add_subdirectory(image)
add_subdirectory(signal)