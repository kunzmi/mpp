					
if (MPP_ENABLE_NPP_BACKEND AND MPP_ENABLE_CUDA_BACKEND AND MPP_ENABLE_CPUSIMPLE_BACKEND)
add_executable(perfTest 
				perfTest.cpp
				runtime.cpp
				testBase.cpp
				pixel8uC1.cpp
				pixel8uC3.cpp
				pixel8uC4.cpp
				pixel16uC1.cpp
				pixel16uC3.cpp
				pixel16uC4.cpp
				pixel32sC1.cpp
				pixel32fC1.cpp
				pixel32fC3.cpp
				pixel32fC4.cpp
				)
				
target_include_directories(perfTest PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(perfTest PUBLIC  
						common
						common.fileIO
						common.image
						backends.cuda	
						backends.cuda.image
						backends.cuda.image.statistics 
						backends.npp
						backends.npp.image
						backends.simple_cpu
)

target_link_libraries(perfTest PRIVATE  
						project_options
						project_warnings)
endif()
