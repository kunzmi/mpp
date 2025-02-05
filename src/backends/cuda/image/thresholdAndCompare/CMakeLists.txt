add_library(backends.cuda.image.thresholdAndCompare 
	compare.cu
	compareEqEps.cu
	threshold.cu
	)
	
target_include_directories(backends.cuda.image.thresholdAndCompare  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.cuda.image.thresholdAndCompare PRIVATE  
						project_options					
						)
target_link_libraries(backends.cuda.image.thresholdAndCompare PUBLIC  
						backends.cuda
						common
						common.image
						)