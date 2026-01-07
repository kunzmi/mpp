add_library(backends.cuda.image.filteringBilateral OBJECT 
	precomputeBilateralGaussFilter.cu)

add_library(backends.cuda.image.filtering8s OBJECT 
	8s/bilateralGaussFilter.cu
	8s/boxAndBoxSquare.cu
	8s/boxFilter.cu
	8s/columnCoefficientFilter.cu
	8s/columnWindowSum.cu
	8s/filter.cu
	8s/fixedSizeBoxFilter.cu
	8s/fixedSizeFilter.cu
	8s/fixedSizeMaxFilter.cu
	8s/fixedSizeMedianFilter.cu
	8s/fixedSizeMinFilter.cu
	8s/fixedSizeSeparableFilter.cu
	8s/fixedSizeUnsharpFilter.cu
	8s/gaussFixed.cu
	8s/gradientVectorPrewitt.cu
	8s/gradientVectorScharr.cu
	8s/gradientVectorSobel.cu
	8s/highpass.cu
	8s/laplace.cu
	8s/lowpass.cu
	8s/maxFilter.cu
	8s/minFilter.cu
	8s/prewittHoriz.cu
	8s/prewittVert.cu
	8s/robertsDown.cu
	8s/robertsUp.cu
	8s/rowCoefficientFilter.cu
	8s/rowWindowSum.cu
	8s/scharrHoriz.cu
	8s/scharrVert.cu
	8s/separableFilter.cu
	8s/sharpen.cu
	8s/sobelCross.cu
	8s/sobelHoriz.cu
	8s/sobelHorizSecond.cu
	8s/sobelVert.cu
	8s/sobelVertSecond.cu
	8s/thresholdAdaptiveBoxFilter.cu
	8s/unsharpFilter.cu
	8s/wienerFilter.cu)

add_library(backends.cuda.image.filtering8u OBJECT 
	8u/bilateralGaussFilter.cu
	8u/boxAndBoxSquare.cu
	8u/boxFilter.cu
	8u/columnCoefficientFilter.cu
	8u/columnWindowSum.cu
	8u/filter.cu
	8u/fixedSizeBoxFilter.cu
	8u/fixedSizeFilter.cu
	8u/fixedSizeMaxFilter.cu
	8u/fixedSizeMedianFilter.cu
	8u/fixedSizeMinFilter.cu
	8u/fixedSizeSeparableFilter.cu
	8u/fixedSizeUnsharpFilter.cu
	8u/gaussFixed.cu
	8u/gradientVectorPrewitt.cu
	8u/gradientVectorScharr.cu
	8u/gradientVectorSobel.cu
	8u/highpass.cu
	8u/laplace.cu
	8u/lowpass.cu
	8u/maxFilter.cu
	8u/minFilter.cu
	8u/prewittHoriz.cu
	8u/prewittVert.cu
	8u/robertsDown.cu
	8u/robertsUp.cu
	8u/rowCoefficientFilter.cu
	8u/rowWindowSum.cu
	8u/scharrHoriz.cu
	8u/scharrVert.cu
	8u/separableFilter.cu
	8u/sharpen.cu
	8u/sobelCross.cu
	8u/sobelHoriz.cu
	8u/sobelHorizSecond.cu
	8u/sobelVert.cu
	8u/sobelVertSecond.cu
	8u/thresholdAdaptiveBoxFilter.cu
	8u/unsharpFilter.cu
	8u/wienerFilter.cu)

add_library(backends.cuda.image.filtering16s OBJECT 
	8u/cannyEdgeHysteresis.cu # final step of a chain, only used for 16s and 32f
	16s/bilateralGaussFilter.cu
	16s/boxAndBoxSquare.cu
	16s/boxFilter.cu
	16s/cannyEdgeMaxSupression.cu
	16s/columnCoefficientFilter.cu
	16s/columnWindowSum.cu
	16s/filter.cu
	16s/fixedSizeBoxFilter.cu
	16s/fixedSizeFilter.cu
	16s/fixedSizeMaxFilter.cu
	16s/fixedSizeMedianFilter.cu
	16s/fixedSizeMinFilter.cu
	16s/fixedSizeSeparableFilter.cu
	16s/fixedSizeUnsharpFilter.cu
	16s/gaussFixed.cu
	16s/gradientVectorPrewitt.cu
	16s/gradientVectorScharr.cu
	16s/gradientVectorSobel.cu
	16s/highpass.cu
	16s/laplace.cu
	16s/lowpass.cu
	16s/maxFilter.cu
	16s/minFilter.cu
	16s/prewittHoriz.cu
	16s/prewittVert.cu
	16s/robertsDown.cu
	16s/robertsUp.cu
	16s/rowCoefficientFilter.cu
	16s/rowWindowSum.cu
	16s/scharrHoriz.cu
	16s/scharrVert.cu
	16s/separableFilter.cu
	16s/sharpen.cu
	16s/sobelCross.cu
	16s/sobelHoriz.cu
	16s/sobelHorizSecond.cu
	16s/sobelVert.cu
	16s/sobelVertSecond.cu
	16s/thresholdAdaptiveBoxFilter.cu
	16s/unsharpFilter.cu
	16s/wienerFilter.cu)

add_library(backends.cuda.image.filtering16u OBJECT 
	16u/bilateralGaussFilter.cu
	16u/boxAndBoxSquare.cu
	16u/boxFilter.cu
	16u/columnCoefficientFilter.cu
	16u/columnWindowSum.cu
	16u/filter.cu
	16u/fixedSizeBoxFilter.cu
	16u/fixedSizeFilter.cu
	16u/fixedSizeMaxFilter.cu
	16u/fixedSizeMedianFilter.cu
	16u/fixedSizeMinFilter.cu
	16u/fixedSizeSeparableFilter.cu
	16u/fixedSizeUnsharpFilter.cu
	16u/gaussFixed.cu
	16u/gradientVectorPrewitt.cu
	16u/gradientVectorScharr.cu
	16u/gradientVectorSobel.cu
	16u/highpass.cu
	16u/laplace.cu
	16u/lowpass.cu
	16u/maxFilter.cu
	16u/minFilter.cu
	16u/prewittHoriz.cu
	16u/prewittVert.cu
	16u/robertsDown.cu
	16u/robertsUp.cu
	16u/rowCoefficientFilter.cu
	16u/rowWindowSum.cu
	16u/scharrHoriz.cu
	16u/scharrVert.cu
	16u/separableFilter.cu
	16u/sharpen.cu
	16u/sobelCross.cu
	16u/sobelHoriz.cu
	16u/sobelHorizSecond.cu
	16u/sobelVert.cu
	16u/sobelVertSecond.cu
	16u/thresholdAdaptiveBoxFilter.cu
	16u/unsharpFilter.cu
	16u/wienerFilter.cu)

add_library(backends.cuda.image.filtering32s OBJECT 
	32s/bilateralGaussFilter.cu
	32s/boxAndBoxSquare.cu
	32s/boxFilter.cu
	32s/columnCoefficientFilter.cu
	32s/columnWindowSum.cu
	32s/filter.cu
	32s/fixedSizeBoxFilter.cu
	32s/fixedSizeFilter.cu
	32s/fixedSizeMaxFilter.cu
	32s/fixedSizeMedianFilter.cu
	32s/fixedSizeMinFilter.cu
	32s/fixedSizeSeparableFilter.cu
	32s/fixedSizeUnsharpFilter.cu
	32s/gaussFixed.cu
	32s/highpass.cu
	32s/laplace.cu
	32s/lowpass.cu
	32s/maxFilter.cu
	32s/minFilter.cu
	32s/prewittHoriz.cu
	32s/prewittVert.cu
	32s/robertsDown.cu
	32s/robertsUp.cu
	32s/rowCoefficientFilter.cu
	32s/rowWindowSum.cu
	32s/scharrHoriz.cu
	32s/scharrVert.cu
	32s/separableFilter.cu
	32s/sharpen.cu
	32s/sobelCross.cu
	32s/sobelHoriz.cu
	32s/sobelHorizSecond.cu
	32s/sobelVert.cu
	32s/sobelVertSecond.cu
	32s/thresholdAdaptiveBoxFilter.cu
	32s/unsharpFilter.cu
	32s/wienerFilter.cu)

add_library(backends.cuda.image.filtering32u OBJECT 
	32u/bilateralGaussFilter.cu
	32u/boxAndBoxSquare.cu
	32u/boxFilter.cu
	32u/columnCoefficientFilter.cu
	32u/columnWindowSum.cu
	32u/filter.cu
	32u/fixedSizeBoxFilter.cu
	32u/fixedSizeFilter.cu
	32u/fixedSizeMaxFilter.cu
	32u/fixedSizeMedianFilter.cu
	32u/fixedSizeMinFilter.cu
	32u/fixedSizeSeparableFilter.cu
	32u/fixedSizeUnsharpFilter.cu
	32u/gaussFixed.cu
	32u/highpass.cu
	32u/laplace.cu
	32u/lowpass.cu
	32u/maxFilter.cu
	32u/minFilter.cu
	32u/prewittHoriz.cu
	32u/prewittVert.cu
	32u/robertsDown.cu
	32u/robertsUp.cu
	32u/rowCoefficientFilter.cu
	32u/rowWindowSum.cu
	32u/scharrHoriz.cu
	32u/scharrVert.cu
	32u/separableFilter.cu
	32u/sharpen.cu
	32u/sobelCross.cu
	32u/sobelHoriz.cu
	32u/sobelHorizSecond.cu
	32u/sobelVert.cu
	32u/sobelVertSecond.cu
	32u/thresholdAdaptiveBoxFilter.cu
	32u/unsharpFilter.cu
	32u/wienerFilter.cu)

add_library(backends.cuda.image.filtering16f OBJECT 
	16f/bilateralGaussFilter.cu
	16f/boxAndBoxSquare.cu
	16f/boxFilter.cu
	16f/columnCoefficientFilter.cu
	16f/columnWindowSum.cu
	16f/filter.cu
	16f/fixedSizeBoxFilter.cu
	16f/fixedSizeFilter.cu
	16f/fixedSizeMaxFilter.cu
	16f/fixedSizeMedianFilter.cu
	16f/fixedSizeMinFilter.cu
	16f/fixedSizeSeparableFilter.cu
	16f/fixedSizeUnsharpFilter.cu
	16f/gaussFixed.cu
	16f/gradientVectorPrewitt.cu
	16f/gradientVectorScharr.cu
	16f/gradientVectorSobel.cu
	16f/highpass.cu
	16f/laplace.cu
	16f/lowpass.cu
	16f/maxFilter.cu
	16f/minFilter.cu
	16f/prewittHoriz.cu
	16f/prewittVert.cu
	16f/robertsDown.cu
	16f/robertsUp.cu
	16f/rowCoefficientFilter.cu
	16f/rowWindowSum.cu
	16f/scharrHoriz.cu
	16f/scharrVert.cu
	16f/separableFilter.cu
	16f/sharpen.cu
	16f/sobelCross.cu
	16f/sobelHoriz.cu
	16f/sobelHorizSecond.cu
	16f/sobelVert.cu
	16f/sobelVertSecond.cu
	16f/thresholdAdaptiveBoxFilter.cu
	16f/unsharpFilter.cu
	16f/wienerFilter.cu)

add_library(backends.cuda.image.filtering16bf OBJECT 
	16bf/bilateralGaussFilter.cu
	16bf/boxAndBoxSquare.cu
	16bf/boxFilter.cu
	16bf/columnCoefficientFilter.cu
	16bf/columnWindowSum.cu
	16bf/filter.cu
	16bf/fixedSizeBoxFilter.cu
	16bf/fixedSizeFilter.cu
	16bf/fixedSizeMaxFilter.cu
	16bf/fixedSizeMedianFilter.cu
	16bf/fixedSizeMinFilter.cu
	16bf/fixedSizeSeparableFilter.cu
	16bf/fixedSizeUnsharpFilter.cu
	16bf/gaussFixed.cu
	16bf/gradientVectorPrewitt.cu
	16bf/gradientVectorScharr.cu
	16bf/gradientVectorSobel.cu
	16bf/highpass.cu
	16bf/laplace.cu
	16bf/lowpass.cu
	16bf/maxFilter.cu
	16bf/minFilter.cu
	16bf/prewittHoriz.cu
	16bf/prewittVert.cu
	16bf/robertsDown.cu
	16bf/robertsUp.cu
	16bf/rowCoefficientFilter.cu
	16bf/rowWindowSum.cu
	16bf/scharrHoriz.cu
	16bf/scharrVert.cu
	16bf/separableFilter.cu
	16bf/sharpen.cu
	16bf/sobelCross.cu
	16bf/sobelHoriz.cu
	16bf/sobelHorizSecond.cu
	16bf/sobelVert.cu
	16bf/sobelVertSecond.cu
	16bf/thresholdAdaptiveBoxFilter.cu
	16bf/unsharpFilter.cu
	16bf/wienerFilter.cu)

add_library(backends.cuda.image.filtering32f OBJECT 
	8u/cannyEdgeHysteresis.cu # final step of a chain, only used for 16s and 32f
	32f/bilateralGaussFilter.cu
	32f/boxAndBoxSquare.cu
	32f/boxFilter.cu
	32f/cannyEdgeMaxSupression.cu
	32f/columnCoefficientFilter.cu
	32f/columnWindowSum.cu
	32f/filter.cu
	32f/fixedSizeBoxFilter.cu
	32f/fixedSizeFilter.cu
	32f/fixedSizeMaxFilter.cu
	32f/fixedSizeMedianFilter.cu
	32f/fixedSizeMinFilter.cu
	32f/fixedSizeSeparableFilter.cu
	32f/fixedSizeUnsharpFilter.cu
	32f/gaussFixed.cu
	32f/gradientVectorPrewitt.cu
	32f/gradientVectorScharr.cu
	32f/gradientVectorSobel.cu
	32f/harrisCornerResponse.cu
	32f/highpass.cu
	32f/laplace.cu
	32f/lowpass.cu
	32f/maxFilter.cu
	32f/minFilter.cu
	32f/prewittHoriz.cu
	32f/prewittVert.cu
	32f/robertsDown.cu
	32f/robertsUp.cu
	32f/rowCoefficientFilter.cu
	32f/rowWindowSum.cu
	32f/scharrHoriz.cu
	32f/scharrVert.cu
	32f/separableFilter.cu
	32f/sharpen.cu
	32f/sobelCross.cu
	32f/sobelHoriz.cu
	32f/sobelHorizSecond.cu
	32f/sobelVert.cu
	32f/sobelVertSecond.cu
	32f/thresholdAdaptiveBoxFilter.cu
	32f/unsharpFilter.cu
	32f/wienerFilter.cu)

add_library(backends.cuda.image.filtering64f OBJECT 
	64f/bilateralGaussFilter.cu
	64f/boxAndBoxSquare.cu
	64f/boxFilter.cu
	64f/columnCoefficientFilter.cu
	64f/columnWindowSum.cu
	64f/filter.cu
	64f/fixedSizeBoxFilter.cu
	64f/fixedSizeFilter.cu
	64f/fixedSizeMaxFilter.cu
	64f/fixedSizeMedianFilter.cu
	64f/fixedSizeMinFilter.cu
	64f/fixedSizeSeparableFilter.cu
	64f/fixedSizeUnsharpFilter.cu
	64f/gaussFixed.cu
	64f/highpass.cu
	64f/laplace.cu
	64f/lowpass.cu
	64f/maxFilter.cu
	64f/minFilter.cu
	64f/prewittHoriz.cu
	64f/prewittVert.cu
	64f/robertsDown.cu
	64f/robertsUp.cu
	64f/rowCoefficientFilter.cu
	64f/rowWindowSum.cu
	64f/scharrHoriz.cu
	64f/scharrVert.cu
	64f/separableFilter.cu
	64f/sharpen.cu
	64f/sobelCross.cu
	64f/sobelHoriz.cu
	64f/sobelHorizSecond.cu
	64f/sobelVert.cu
	64f/sobelVertSecond.cu
	64f/thresholdAdaptiveBoxFilter.cu
	64f/unsharpFilter.cu
	64f/wienerFilter.cu)

add_library(backends.cuda.image.filtering16sc OBJECT 
	16sc/boxFilter.cu
	16sc/columnCoefficientFilter.cu
	16sc/columnWindowSum.cu
	16sc/filter.cu
	16sc/fixedSizeBoxFilter.cu
	16sc/fixedSizeFilter.cu
	16sc/fixedSizeSeparableFilter.cu
	16sc/gaussFixed.cu
	16sc/highpass.cu
	16sc/laplace.cu
	16sc/lowpass.cu
	16sc/prewittHoriz.cu
	16sc/prewittVert.cu
	16sc/robertsDown.cu
	16sc/robertsUp.cu
	16sc/rowCoefficientFilter.cu
	16sc/rowWindowSum.cu
	16sc/scharrHoriz.cu
	16sc/scharrVert.cu
	16sc/separableFilter.cu
	16sc/sharpen.cu
	16sc/sobelCross.cu
	16sc/sobelHoriz.cu
	16sc/sobelHorizSecond.cu
	16sc/sobelVert.cu
	16sc/sobelVertSecond.cu)

add_library(backends.cuda.image.filtering32sc OBJECT
	32sc/boxFilter.cu
	32sc/columnCoefficientFilter.cu
	32sc/columnWindowSum.cu
	32sc/filter.cu
	32sc/fixedSizeBoxFilter.cu
	32sc/fixedSizeFilter.cu
	32sc/fixedSizeSeparableFilter.cu
	32sc/gaussFixed.cu
	32sc/highpass.cu
	32sc/laplace.cu
	32sc/lowpass.cu
	32sc/prewittHoriz.cu
	32sc/prewittVert.cu
	32sc/robertsDown.cu
	32sc/robertsUp.cu
	32sc/rowCoefficientFilter.cu
	32sc/rowWindowSum.cu
	32sc/scharrHoriz.cu
	32sc/scharrVert.cu
	32sc/separableFilter.cu
	32sc/sharpen.cu
	32sc/sobelCross.cu
	32sc/sobelHoriz.cu
	32sc/sobelHorizSecond.cu
	32sc/sobelVert.cu
	32sc/sobelVertSecond.cu)

add_library(backends.cuda.image.filtering32fc OBJECT
	32fc/boxFilter.cu
	32fc/columnCoefficientFilter.cu
	32fc/columnWindowSum.cu
	32fc/filter.cu
	32fc/fixedSizeBoxFilter.cu
	32fc/fixedSizeFilter.cu
	32fc/fixedSizeSeparableFilter.cu
	32fc/gaussFixed.cu
	32fc/highpass.cu
	32fc/laplace.cu
	32fc/lowpass.cu
	32fc/prewittHoriz.cu
	32fc/prewittVert.cu
	32fc/robertsDown.cu
	32fc/robertsUp.cu
	32fc/rowCoefficientFilter.cu
	32fc/rowWindowSum.cu
	32fc/scharrHoriz.cu
	32fc/scharrVert.cu
	32fc/separableFilter.cu
	32fc/sharpen.cu
	32fc/sobelCross.cu
	32fc/sobelHoriz.cu
	32fc/sobelHorizSecond.cu
	32fc/sobelVert.cu
	32fc/sobelVertSecond.cu
	
	)
	
set_source_files_properties(8s/bilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(8u/bilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(16s/bilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(16u/bilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(32s/bilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(32u/bilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(16f/bilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(16bf/bilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(32f/bilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(64f/bilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(precomputeBilateralGaussFilter.cu PROPERTIES COMPILE_FLAGS --use_fast_math)

	
target_include_directories(backends.cuda.image.filtering8s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering8s PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering8u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering8u PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering16s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering16s PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering16u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering16u PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering32s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering32s PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering32u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering32u PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering16sc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering16sc PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering32sc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering32sc PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering16bf PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering16bf PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering16f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering16f PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering32f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering32f PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering32fc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering32fc PRIVATE project_options)

target_include_directories(backends.cuda.image.filtering64f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filtering64f PRIVATE project_options)

target_include_directories(backends.cuda.image.filteringBilateral PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.filteringBilateral PRIVATE project_options)
