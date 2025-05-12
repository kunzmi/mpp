add_library(backends.cuda.image.geometryTransforms 
	affineBack_part1.cu
	affineBack_part2.cu
	affineBack_part3.cu
	affineBack_part4.cu
	affineBack_P2.cu
	affineBack_P3.cu
	affineBack_P4.cu

	mirror.cu
	
	perspectiveBack_part1.cu
	perspectiveBack_part2.cu
	perspectiveBack_part3.cu
	perspectiveBack_part4.cu
	perspectiveBack_P2.cu
	perspectiveBack_P3.cu
	perspectiveBack_P4.cu
	
	remap_part1.cu
	remap_part2.cu
	remap_part3.cu
	remap_part4.cu
	remap_P2.cu
	remap_P3.cu
	remap_P4.cu
	
	resize_part1.cu
	resize_part2.cu
	resize_part3.cu
	resize_part4.cu
	resize_P2.cu
	resize_P3.cu
	resize_P4.cu
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