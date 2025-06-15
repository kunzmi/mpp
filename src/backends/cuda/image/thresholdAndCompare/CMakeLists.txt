add_library(backends.cuda.image.thresholdAndCompare 
	8s/compare.cu
	8s/threshold.cu
	
	8u/compare.cu
	8u/threshold.cu
	
	16s/compare.cu
	16s/threshold.cu
	
	16u/compare.cu
	16u/threshold.cu
	
	16f/compare.cu
	16f/compareEqEps.cu
	16f/threshold.cu
	
	16bf/compare.cu
	16bf/compareEqEps.cu
	16bf/threshold.cu
	
	32s/compare.cu
	32s/threshold.cu
	
	32u/compare.cu
	32u/threshold.cu
	
	32f/compare.cu
	32f/compareEqEps.cu
	32f/threshold.cu
	
	64f/compare.cu
	64f/compareEqEps.cu
	64f/threshold.cu
	
	
	16sc/compare.cu
	
	32sc/compare.cu
	
	32fc/compare.cu
	32fc/compareEqEps.cu
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