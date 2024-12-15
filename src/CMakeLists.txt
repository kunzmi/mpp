
add_subdirectory(backends)
add_subdirectory(common)
add_subdirectory(utilities)

add_executable(dummy_test 
				main.cpp
				)


target_include_directories(dummy_test PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(dummy_test PUBLIC  

						common
						common.fileIO
						common.image
						backends.cuda
						backends.cuda.image
						backends.npp
						backends.npp.image)
target_link_libraries(dummy_test PRIVATE  
						project_options
						project_warnings)

						

