add_library(backends.cuda.image.dataExchangeAndInit 
	convert.cu

	scale.cu

	set.cu
	setMasked.cu

	)
	
target_include_directories(backends.cuda.image.dataExchangeAndInit  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.cuda.image.dataExchangeAndInit PRIVATE  
						project_options					
						)
target_link_libraries(backends.cuda.image.dataExchangeAndInit PUBLIC  
						backends.cuda
						common
						common.image
						)