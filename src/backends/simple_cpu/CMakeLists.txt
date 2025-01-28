if (OPP_ENABLE_CPUSIMPLE_BACKEND)

add_library(backends.simple_cpu 
			image/imageView.cpp
)
	
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
						)
endif()
