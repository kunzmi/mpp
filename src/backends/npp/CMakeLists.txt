add_library(backends.npp 
	nppException.cpp
	)
	
target_include_directories(backends.npp  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.npp PRIVATE  
						project_options
						project_warnings						
						)
target_link_libraries(backends.npp PUBLIC  
						backends.cuda
						backends.cuda.image
						common
						common.image 
						
						CUDA::nppc
						CUDA::nppial
						CUDA::nppicc
						CUDA::nppidei
						CUDA::nppif
						CUDA::nppig
						CUDA::nppim
						CUDA::nppist
						CUDA::nppisu
						CUDA::nppitc
						CUDA::npps
						)

add_subdirectory(image)