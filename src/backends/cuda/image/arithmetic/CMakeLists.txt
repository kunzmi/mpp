add_library(backends.cuda.image.arithmetic 
	#addSrcSrcKernel.cu
	add.cu
	addMasked.cu
	
	set.cu
	setMasked.cu
	)
	
target_include_directories(backends.cuda.image.arithmetic  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.cuda.image.arithmetic PRIVATE  
						project_options
						#project_warnings						
						)
target_link_libraries(backends.cuda.image.arithmetic PUBLIC  
						backends.cuda
						common
						common.image
						)