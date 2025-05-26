add_library(backends.cuda.image.filtering 
	bilateralGaussFilter.cu
	precomputeBilateralGaussFilter.cu
	boxAndBoxSquare.cu
	boxFilter.cu
	cannyEdgeHysteresis.cu
	cannyEdgeMaxSupression.cu
	columnCoefficientFilter.cu
	columnWindowSum.cu
	filter.cu
	fixedSizeBoxFilter.cu
	fixedSizeFilter.cu
	fixedSizeMaxFilter.cu
	fixedSizeMinFilter.cu
	fixedSizeSeparableFilter.cu
	fixedSizeUnsharpFilter.cu
	gaussFixed_part1.cu
	gaussFixed_part2.cu
	gaussFixed_part3.cu
	gaussFixed_part4.cu
	gradientVectorPrewitt.cu
	gradientVectorScharr.cu
	gradientVectorSobel.cu
	harrisCornerResponse.cu
	highpass.cu
	laplace.cu
	lowpass.cu
	maxFilter.cu
	minFilter.cu
	prewittHoriz.cu
	prewittVert.cu
	robertsDown.cu
	robertsUp.cu
	rowCoefficientFilter.cu
	rowWindowSum.cu
	scharrHoriz.cu
	scharrVert.cu
	separableFilter.cu
	sharpen.cu
	sobelCross.cu
	sobelHoriz.cu
	sobelHorizSecond.cu
	sobelVert.cu
	sobelVertSecond.cu
	thresholdAdaptiveBoxFilter.cu
	unsharpFilter.cu
	wienerFilter.cu
	)
	
	set_source_files_properties(bilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
	set_source_files_properties(precomputeBilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)

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