if (MPP_ENABLE_CPUSIMPLE_BACKEND)

add_library(backends.simple_cpu8u SHARED image/imageView8u.cpp)
target_compile_definitions(backends.simple_cpu8u PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu8u  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu8u PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu8u PUBLIC common)

add_library(backends.simple_cpu8s SHARED image/imageView8s.cpp)
target_compile_definitions(backends.simple_cpu8s PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu8s  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu8s PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu8s PUBLIC common)


add_library(backends.simple_cpu16u SHARED image/imageView16u.cpp)
target_compile_definitions(backends.simple_cpu16u PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu16u  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu16u PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu16u PUBLIC common)

add_library(backends.simple_cpu16s SHARED image/imageView16s.cpp)
target_compile_definitions(backends.simple_cpu16s PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu16s  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu16s PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu16s PUBLIC common)


add_library(backends.simple_cpu32u SHARED image/imageView32u.cpp)
target_compile_definitions(backends.simple_cpu32u PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu32u  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu32u PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu32u PUBLIC common)

add_library(backends.simple_cpu32s SHARED image/imageView32s.cpp)
target_compile_definitions(backends.simple_cpu32s PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu32s  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu32s PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu32s PUBLIC common)


add_library(backends.simple_cpu16bf SHARED image/imageView16bf.cpp)
target_compile_definitions(backends.simple_cpu16bf PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu16bf  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu16bf PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu16bf PUBLIC common)

add_library(backends.simple_cpu16f SHARED image/imageView16f.cpp)
target_compile_definitions(backends.simple_cpu16f PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu16f  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu16f PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu16f PUBLIC common)

add_library(backends.simple_cpu32f SHARED image/imageView32f.cpp)
target_compile_definitions(backends.simple_cpu32f PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu32f  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu32f PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu32f PUBLIC common)

add_library(backends.simple_cpu64f SHARED image/imageView64f.cpp)
target_compile_definitions(backends.simple_cpu64f PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu64f  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu64f PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu64f PUBLIC common)


add_library(backends.simple_cpu16sc SHARED image/imageView16sc.cpp)
target_compile_definitions(backends.simple_cpu16sc PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu16sc  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu16sc PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu16sc PUBLIC common)

add_library(backends.simple_cpu32sc SHARED image/imageView32sc.cpp)
target_compile_definitions(backends.simple_cpu32sc PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu32sc  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu32sc PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu32sc PUBLIC common)

add_library(backends.simple_cpu32fc SHARED image/imageView32fc.cpp)
target_compile_definitions(backends.simple_cpu32fc PUBLIC -DMPP_SIMPLECPU)	
target_include_directories(backends.simple_cpu32fc  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.simple_cpu32fc PRIVATE project_options project_warnings)
target_link_libraries(backends.simple_cpu32fc PUBLIC common)


set_target_properties(backends.simple_cpu8u PROPERTIES OUTPUT_NAME mppi-simplecpu-8u)
set_target_properties(backends.simple_cpu8s PROPERTIES OUTPUT_NAME mppi-simplecpu-8s)
set_target_properties(backends.simple_cpu16u PROPERTIES OUTPUT_NAME mppi-simplecpu-16u)
set_target_properties(backends.simple_cpu16s PROPERTIES OUTPUT_NAME mppi-simplecpu-16s)
set_target_properties(backends.simple_cpu32u PROPERTIES OUTPUT_NAME mppi-simplecpu-32u)
set_target_properties(backends.simple_cpu32s PROPERTIES OUTPUT_NAME mppi-simplecpu-32s)
set_target_properties(backends.simple_cpu16f PROPERTIES OUTPUT_NAME mppi-simplecpu-16f)
set_target_properties(backends.simple_cpu16bf PROPERTIES OUTPUT_NAME mppi-simplecpu-16bf)
set_target_properties(backends.simple_cpu32f PROPERTIES OUTPUT_NAME mppi-simplecpu-32f)
set_target_properties(backends.simple_cpu64f PROPERTIES OUTPUT_NAME mppi-simplecpu-64f)
set_target_properties(backends.simple_cpu16sc PROPERTIES OUTPUT_NAME mppi-simplecpu-16sc)
set_target_properties(backends.simple_cpu32sc PROPERTIES OUTPUT_NAME mppi-simplecpu-32sc)
set_target_properties(backends.simple_cpu32fc PROPERTIES OUTPUT_NAME mppi-simplecpu-32fc)

add_library(backends.simple_cpu INTERFACE)
target_link_libraries(backends.simple_cpu INTERFACE 
		backends.simple_cpu8u
		backends.simple_cpu8s
		backends.simple_cpu16u
		backends.simple_cpu16s
		backends.simple_cpu32u
		backends.simple_cpu32s
		
		backends.simple_cpu16bf
		backends.simple_cpu16f
		backends.simple_cpu32f
		backends.simple_cpu64f

		backends.simple_cpu16sc
		backends.simple_cpu32sc
		backends.simple_cpu32fc
)
endif()
