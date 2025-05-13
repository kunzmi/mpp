add_library(backends.cuda.image.filtering 
	gaussFixed_part1.cu
	gaussFixed_part2.cu
	gaussFixed_part3.cu
	gaussFixed_part4.cu
	highpass.cu
	laplace.cu
	lowpass.cu
	prewittHoriz.cu
	prewittVert.cu
	robertsDown.cu
	robertsUp.cu
	scharrHoriz.cu
	scharrVert.cu
	sharpen.cu
	sobelCross.cu
	sobelHoriz.cu
	sobelHorizSecond.cu
	sobelVert.cu
	sobelVertSecond.cu
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