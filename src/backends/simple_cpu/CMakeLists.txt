if (MPP_ENABLE_CPUSIMPLE_BACKEND)

add_library(backends.simple_cpu SHARED
			image/imageView.cpp
)

target_compile_definitions(backends.simple_cpu PUBLIC -DMPP_SIMPLECPU)  
	
target_include_directories(backends.simple_cpu  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.simple_cpu PRIVATE  
						project_options
						project_warnings						
						)
target_link_libraries(backends.simple_cpu PUBLIC  
						common
						backends.npp
						backends.cuda.image8u
						backends.cuda.image8s
						backends.cuda.image16u
						backends.cuda.image16s
						backends.cuda.image32u
						backends.cuda.image32s
						backends.cuda.image16f
						backends.cuda.image16bf
						backends.cuda.image32f
						backends.cuda.image64f
						backends.cuda.image16sc
						backends.cuda.image32sc
						backends.cuda.image32fc
						CUDA::cudart
						)
endif()
