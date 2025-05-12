add_library(backends.cuda.image.filtering 
	gaussFixed.cu
	highpass.cu
	)
	
target_include_directories(backends.cuda.image.filtering  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.cuda.image.filtering PRIVATE  
						project_options					
						)
target_link_libraries(backends.cuda.image.filtering PUBLIC  
						backends.cuda
						common
						common.image
						)