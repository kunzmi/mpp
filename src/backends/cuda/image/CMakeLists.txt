add_library(backends.cuda.image 
	forEachPixelKernel.cu
	testKernel.cu
	)
	
target_include_directories(backends.cuda.image  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(backends.cuda.image PRIVATE  
						project_options
						#project_warnings						
						)
target_link_libraries(backends.cuda.image PUBLIC  
						common
						common.image
						)