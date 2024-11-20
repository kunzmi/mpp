
add_subdirectory(backends)
add_subdirectory(common)
add_subdirectory(utilities)

add_executable(dummy_test 
				main.cpp
				)


target_link_libraries(dummy_test PUBLIC  
						common
						common.image
						backends.cuda.image)
target_link_libraries(dummy_test PRIVATE  
						project_options
						project_warnings)

						

