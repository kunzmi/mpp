
add_subdirectory(arithmetic)
add_subdirectory(dataExchangeAndInit)
add_subdirectory(filtering)
add_subdirectory(geometryTransforms)
add_subdirectory(statistics)
add_subdirectory(thresholdAndCompare)

add_library(backends.cuda.image 
						imageView.cpp
						)

target_include_directories(backends.cuda.image PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.cuda.image PRIVATE  
						project_options					
						)
target_link_libraries(backends.cuda.image PUBLIC  
						backends.cuda.image.arithmetic
						backends.cuda.image.dataExchangeAndInit 
						backends.cuda.image.filtering 
						backends.cuda.image.geometryTransforms 
						backends.cuda.image.statistics 
						backends.cuda.image.thresholdAndCompare 
						backends.cuda
						common
						common.image
						)