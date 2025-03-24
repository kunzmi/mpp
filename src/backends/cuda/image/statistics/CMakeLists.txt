add_library(backends.cuda.image.statistics 
	countInRange.cu
	mean.cu
	meanStd.cu
	max.cu
	min.cu
	maxIdx.cu
	minIdx.cu
	minMax.cu
	minMaxEvery.cu
	minMaxIdx.cu
	normInf.cu
	normL1.cu
	normL2.cu
	sum.cu
	)
	
target_include_directories(backends.cuda.image.statistics  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.cuda.image.statistics PRIVATE  
						project_options					
						)
target_link_libraries(backends.cuda.image.statistics PUBLIC  
						backends.cuda
						common
						common.image
						)