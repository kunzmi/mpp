if (MPP_ENABLE_CUDA_CORE)

set(filesForbackends.cuda 
	cudaException.cpp
	event.cpp
	stream.cpp
)
if(MPP_ENABLE_CUDA_BACKEND)
	set(filesForbackends.cuda ${filesForbackends.cuda} streamCtx.cpp)
endif()

add_library(backends.cuda ${filesForbackends.cuda})
	
target_include_directories(backends.cuda  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.cuda PRIVATE  
						project_options
						project_warnings						
						)
target_link_libraries(backends.cuda PUBLIC  
						common
						)
endif()

if(MPP_ENABLE_CUDA_BACKEND)
add_subdirectory(image)
endif()