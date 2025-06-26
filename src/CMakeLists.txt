
add_subdirectory(backends)
add_subdirectory(common)
add_subdirectory(utilities)

add_executable(dummy_test 
				main.cpp
				)

set(linkListForDummyTest 
						common
						common.fileIO
						common.image
)
if(MPP_ENABLE_CUDA_CORE)
	set(linkListForDummyTest ${linkListForDummyTest} backends.cuda)
endif()
if(MPP_ENABLE_CUDA_BACKEND)
	set(linkListForDummyTest ${linkListForDummyTest} backends.cuda.image)
endif()
if(MPP_ENABLE_NPP_BACKEND)
	set(linkListForDummyTest ${linkListForDummyTest} 
						#backends.npp
						#backends.npp.image
	)
endif()
if(MPP_ENABLE_CPUSIMPLE_BACKEND)
	set(linkListForDummyTest ${linkListForDummyTest} backends.simple_cpu)
endif()

target_include_directories(dummy_test PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(dummy_test PUBLIC  ${linkListForDummyTest})

target_link_libraries(dummy_test PRIVATE  
						project_options
						project_warnings)

						
#if (MPP_ENABLE_NPP_BACKEND AND MPP_ENABLE_CUDA_BACKEND AND MPP_ENABLE_CPUSIMPLE_BACKEND)
#add_executable(perfTest 
#				perfTest.cpp
#				)
#				
#target_include_directories(perfTest PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#
#target_link_libraries(perfTest PUBLIC  
#						common
#						common.fileIO
#						common.image
#						backends.cuda	
#						backends.cuda.image
#						backends.cuda.image.statistics 
#						backends.npp
#						backends.npp.image
#						backends.simple_cpu
#)
#
#target_link_libraries(perfTest PRIVATE  
#						project_options
#						project_warnings)
#endif()
