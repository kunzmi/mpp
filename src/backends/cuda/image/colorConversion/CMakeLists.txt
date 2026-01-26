
	
add_library(backends.cuda.image.colorConversion8u OBJECT 
	8u/hls.cu
	8u/hsv.cu
	8u/lab.cu
	8u/luv.cu
	
	8u/colorTwist3x3.cu
	8u/colorTwist3x4.cu
	8u/colorTwist4x4.cu
	8u/colorTwist4x4C.cu
	
	8u/colorGradientToGray.cu
	8u/colorToGray.cu
	8u/gammaCorr.cu
	
	8u/cfaToRgb.cu
	8u/rgbToCfa.cu

	8u/lut.cu
)
	
target_include_directories(backends.cuda.image.colorConversion8u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.colorConversion8u PRIVATE project_options)

add_library(backends.cuda.image.colorConversion16u OBJECT 
	16u/hls.cu
	16u/hsv.cu
	16u/lab.cu
	16u/luv.cu
	
	16u/colorTwist3x3.cu
	16u/colorTwist3x4.cu
	16u/colorTwist4x4.cu
	16u/colorTwist4x4C.cu
	
	16u/colorGradientToGray.cu
	16u/colorToGray.cu
	16u/gammaCorr.cu
	
	16u/cfaToRgb.cu
	16u/rgbToCfa.cu

	16u/lut.cu
)
	
target_include_directories(backends.cuda.image.colorConversion16u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.colorConversion16u PRIVATE project_options)

add_library(backends.cuda.image.colorConversion16s OBJECT 
	16s/colorTwist3x3.cu
	16s/colorTwist3x4.cu
	16s/colorTwist4x4.cu
	16s/colorTwist4x4C.cu
	  
	16s/colorGradientToGray.cu
	16s/colorToGray.cu
	16s/gammaCorr.cu
	
	16s/cfaToRgb.cu
	16s/rgbToCfa.cu

	16s/lut.cu
)
	
target_include_directories(backends.cuda.image.colorConversion16s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.colorConversion16s PRIVATE project_options)

add_library(backends.cuda.image.colorConversion16f OBJECT 
	16f/hls.cu
	16f/hsv.cu
	16f/lab.cu
	16f/luv.cu
	
	16f/colorTwist3x3.cu
	16f/colorTwist3x4.cu
	16f/colorTwist4x4.cu
	16f/colorTwist4x4C.cu
	
	16f/colorGradientToGray.cu
	16f/colorToGray.cu
	16f/gammaCorr.cu
	
	16f/cfaToRgb.cu
	16f/rgbToCfa.cu

	16f/lut.cu
)
	
target_include_directories(backends.cuda.image.colorConversion16f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.colorConversion16f PRIVATE project_options)

add_library(backends.cuda.image.colorConversion32f OBJECT 
	32f/hls.cu
	32f/hsv.cu
	32f/lab.cu
	32f/luv.cu
	
	32f/colorTwist3x3.cu
	32f/colorTwist3x4.cu
	32f/colorTwist4x4.cu
	32f/colorTwist4x4C.cu
	
	32f/colorGradientToGray.cu
	32f/colorToGray.cu
	32f/gammaCorr.cu
	
	32f/cfaToRgb.cu
	32f/rgbToCfa.cu

	32f/lut.cu
)
	
target_include_directories(backends.cuda.image.colorConversion32f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.colorConversion32f PRIVATE project_options)



add_library(backends.cuda.image.colorConversion16bf OBJECT 
	16bf/cfaToRgb.cu
	16bf/rgbToCfa.cu

	16bf/lut.cu
)
	
target_include_directories(backends.cuda.image.colorConversion16bf PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.colorConversion16bf PRIVATE project_options)

add_library(backends.cuda.image.colorConversion32s OBJECT 
	32s/cfaToRgb.cu
	32s/rgbToCfa.cu
)
	
target_include_directories(backends.cuda.image.colorConversion32s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.colorConversion32s PRIVATE project_options)

add_library(backends.cuda.image.colorConversion32u OBJECT 
	32u/cfaToRgb.cu
	32u/rgbToCfa.cu
)
	
target_include_directories(backends.cuda.image.colorConversion32u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.colorConversion32u PRIVATE project_options)
