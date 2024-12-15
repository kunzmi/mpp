add_library(backends.npp.image 
	image8uC1.cpp
	image8uC1View.cpp
	)
	
target_include_directories(backends.npp.image PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.npp.image PRIVATE  
						project_options
						project_warnings						
						)
target_link_libraries(backends.npp.image PUBLIC  
						backends.npp
						)
