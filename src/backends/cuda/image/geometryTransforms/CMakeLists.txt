add_library(backends.cuda.image.geometryTransforms 
	8s/affineBack.cu
	8u/affineBack.cu
	16s/affineBack.cu
	16u/affineBack.cu
	32s/affineBack.cu
	32u/affineBack.cu
	16f/affineBack.cu
	16bf/affineBack.cu
	32f/affineBack.cu
	64f/affineBack.cu
	16sc/affineBack.cu
	32sc/affineBack.cu
	32fc/affineBack.cu
	
	8s/mirror.cu
	8u/mirror.cu
	16s/mirror.cu
	16u/mirror.cu
	32s/mirror.cu
	32u/mirror.cu
	16f/mirror.cu
	16bf/mirror.cu
	32f/mirror.cu
	64f/mirror.cu
	16sc/mirror.cu
	32sc/mirror.cu
	32fc/mirror.cu
	
	8s/perspectiveBack.cu
	8u/perspectiveBack.cu
	16s/perspectiveBack.cu
	16u/perspectiveBack.cu
	32s/perspectiveBack.cu
	32u/perspectiveBack.cu
	16f/perspectiveBack.cu
	16bf/perspectiveBack.cu
	32f/perspectiveBack.cu
	64f/perspectiveBack.cu
	16sc/perspectiveBack.cu
	32sc/perspectiveBack.cu
	32fc/perspectiveBack.cu
	
	8s/remap.cu
	8u/remap.cu
	16s/remap.cu
	16u/remap.cu
	32s/remap.cu
	32u/remap.cu
	16f/remap.cu
	16bf/remap.cu
	32f/remap.cu
	64f/remap.cu
	16sc/remap.cu
	32sc/remap.cu
	32fc/remap.cu

	8s/resize.cu
	8u/resize.cu
	16s/resize.cu
	16u/resize.cu
	32s/resize.cu
	32u/resize.cu
	16f/resize.cu
	16bf/resize.cu
	32f/resize.cu
	64f/resize.cu
	16sc/resize.cu
	32sc/resize.cu
	32fc/resize.cu
	)
	
target_include_directories(backends.cuda.image.geometryTransforms  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.cuda.image.geometryTransforms PRIVATE  
						project_options					
						)
target_link_libraries(backends.cuda.image.geometryTransforms PUBLIC  
						backends.cuda
						common
						common.image
						)