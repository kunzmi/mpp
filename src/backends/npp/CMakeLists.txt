if (MPP_ENABLE_NPP_BACKEND)

add_subdirectory(image)
add_library(backends.npp SHARED
	nppException.cpp
	)

target_compile_definitions(backends.npp PRIVATE -DMPP_NPP)  
	
target_include_directories(backends.npp  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.npp PRIVATE  
						project_options
						project_warnings						
						)
target_link_libraries(backends.npp PUBLIC  
						backends.cuda  # for device variable and cuda stuff
						backends.npp.image
						common
						#common.image 
						
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

set_target_properties(backends.npp PROPERTIES OUTPUT_NAME mpp-npp)

endif() # MPP_ENABLE_NPP_BACKEND