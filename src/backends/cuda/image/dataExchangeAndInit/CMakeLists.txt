add_library(backends.cuda.image.dataExchangeAndInit 
	convert.cu
	
	copy.cu
	copyBorder.cu
	copyMasked.cu
	copySubpix.cu
	
	dup.cu

	scale.cu

	set.cu
	setChannel.cu
	setMasked.cu

	swapChannel.cu

	transpose.cu
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