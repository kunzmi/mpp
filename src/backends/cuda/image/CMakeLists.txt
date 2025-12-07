
add_subdirectory(arithmetic)
add_subdirectory(dataExchangeAndInit)
add_subdirectory(filtering)
add_subdirectory(geometryTransforms)
add_subdirectory(morphology)
add_subdirectory(statistics)
add_subdirectory(thresholdAndCompare)

#8u
add_library(backends.cuda.image8u SHARED
						imageView8u.cpp
						)
						
target_compile_definitions(backends.cuda.image8u PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image8u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image8u PRIVATE project_options)
target_link_libraries(backends.cuda.image8u PUBLIC  
						backends.cuda.image.arithmetic8u
						backends.cuda.image.dataExchangeAndInit8u 
						backends.cuda.image.filtering8u 
						backends.cuda.image.geometryTransforms8u 
						backends.cuda.image.morphology8u 
						backends.cuda.image.statistics8u 
						backends.cuda.image.thresholdAndCompare8u 
						backends.cuda.image.filteringBilateral 
						backends.cuda.image.dataExchangeAndInitTranspose
						backends.cuda
						common
						)



#8s
add_library(backends.cuda.image8s SHARED
						imageView8s.cpp
						)
						
target_compile_definitions(backends.cuda.image8s PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image8s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image8s PRIVATE project_options)
target_link_libraries(backends.cuda.image8s PUBLIC  
						backends.cuda.image.arithmetic8s
						backends.cuda.image.dataExchangeAndInit8s 
						backends.cuda.image.filtering8s 
						backends.cuda.image.geometryTransforms8s 
						backends.cuda.image.morphology8s 
						backends.cuda.image.statistics8s 
						backends.cuda.image.thresholdAndCompare8s 
						backends.cuda.image.filteringBilateral 
						backends.cuda.image.dataExchangeAndInitTranspose
						backends.cuda
						common
						)

						

#16u
add_library(backends.cuda.image16u SHARED
						imageView16u.cpp
						)
						
target_compile_definitions(backends.cuda.image16u PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image16u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image16u PRIVATE project_options)
target_link_libraries(backends.cuda.image16u PUBLIC  
						backends.cuda.image.arithmetic16u
						backends.cuda.image.dataExchangeAndInit16u 
						backends.cuda.image.filtering16u 
						backends.cuda.image.geometryTransforms16u 
						backends.cuda.image.morphology16u 
						backends.cuda.image.statistics16u 
						backends.cuda.image.thresholdAndCompare16u 
						backends.cuda.image.filteringBilateral 
						backends.cuda.image.dataExchangeAndInitTranspose
						backends.cuda
						common
						)



#16s
add_library(backends.cuda.image16s SHARED
						imageView16s.cpp
						)
						
target_compile_definitions(backends.cuda.image16s PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image16s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image16s PRIVATE project_options)
target_link_libraries(backends.cuda.image16s PUBLIC  
						backends.cuda.image.arithmetic16s
						backends.cuda.image.dataExchangeAndInit16s 
						backends.cuda.image.filtering16s 
						backends.cuda.image.geometryTransforms16s 
						backends.cuda.image.morphology16s 
						backends.cuda.image.statistics16s 
						backends.cuda.image.thresholdAndCompare16s 
						backends.cuda.image.filteringBilateral 
						backends.cuda.image.dataExchangeAndInitTranspose
						backends.cuda
						common
						)
						

#32u
add_library(backends.cuda.image32u SHARED
						imageView32u.cpp
						)
						
target_compile_definitions(backends.cuda.image32u PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image32u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image32u PRIVATE project_options)
target_link_libraries(backends.cuda.image32u PUBLIC  
						backends.cuda.image.arithmetic32u
						backends.cuda.image.dataExchangeAndInit32u 
						backends.cuda.image.filtering32u 
						backends.cuda.image.geometryTransforms32u 
						backends.cuda.image.morphology32u 
						backends.cuda.image.statistics32u 
						backends.cuda.image.thresholdAndCompare32u
						backends.cuda.image.dataExchangeAndInitTranspose 
						backends.cuda
						common
						)



#32s
add_library(backends.cuda.image32s SHARED
						imageView32s.cpp
						)
						
target_compile_definitions(backends.cuda.image32s PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image32s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image32s PRIVATE project_options)
target_link_libraries(backends.cuda.image32s PUBLIC  
						backends.cuda.image.arithmetic32s
						backends.cuda.image.dataExchangeAndInit32s 
						backends.cuda.image.filtering32s 
						backends.cuda.image.geometryTransforms32s 
						backends.cuda.image.morphology32s 
						backends.cuda.image.statistics32s 
						backends.cuda.image.thresholdAndCompare32s 
						backends.cuda.image.dataExchangeAndInitTranspose
						backends.cuda
						common
						)
						



#16sc
add_library(backends.cuda.image16sc SHARED
						imageView16sc.cpp
						)
						
target_compile_definitions(backends.cuda.image16sc PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image16sc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image16sc PRIVATE project_options)
target_link_libraries(backends.cuda.image16sc PUBLIC  
						backends.cuda.image.arithmetic16sc
						backends.cuda.image.dataExchangeAndInit16sc 
						backends.cuda.image.filtering16sc 
						backends.cuda.image.geometryTransforms16sc 
						backends.cuda.image.statistics16sc
						backends.cuda.image.thresholdAndCompare16sc 
						backends.cuda
						common
						)



#32sc
add_library(backends.cuda.image32sc SHARED
						imageView32sc.cpp
						)
						
target_compile_definitions(backends.cuda.image32sc PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image32sc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image32sc PRIVATE project_options)
target_link_libraries(backends.cuda.image32sc PUBLIC  
						backends.cuda.image.arithmetic32sc
						backends.cuda.image.dataExchangeAndInit32sc 
						backends.cuda.image.filtering32sc 
						backends.cuda.image.geometryTransforms32sc 
						backends.cuda.image.statistics32sc 
						backends.cuda.image.thresholdAndCompare32sc 
						backends.cuda
						common
						)



#32fc
add_library(backends.cuda.image32fc SHARED
						imageView32fc.cpp
						)
						
target_compile_definitions(backends.cuda.image32fc PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image32fc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image32fc PRIVATE project_options)
target_link_libraries(backends.cuda.image32fc PUBLIC  
						backends.cuda.image.arithmetic32fc
						backends.cuda.image.dataExchangeAndInit32fc 
						backends.cuda.image.filtering32fc 
						backends.cuda.image.geometryTransforms32fc 
						backends.cuda.image.statistics32fc 
						backends.cuda.image.thresholdAndCompare32fc 
						backends.cuda
						common
						)



#16bf
add_library(backends.cuda.image16bf SHARED
						imageView16bf.cpp
						)
						
target_compile_definitions(backends.cuda.image16bf PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image16bf PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image16bf PRIVATE project_options)
target_link_libraries(backends.cuda.image16bf PUBLIC  
						backends.cuda.image.arithmetic16bf
						backends.cuda.image.dataExchangeAndInit16bf 
						backends.cuda.image.filtering16bf 
						backends.cuda.image.geometryTransforms16bf 
						backends.cuda.image.morphology16bf 
						backends.cuda.image.statistics16bf 
						backends.cuda.image.thresholdAndCompare16bf 
						backends.cuda.image.filteringBilateral 
						backends.cuda.image.dataExchangeAndInitTranspose
						backends.cuda
						common
						)



#16f
add_library(backends.cuda.image16f SHARED
						imageView16f.cpp
						)
						
target_compile_definitions(backends.cuda.image16f PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image16f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image16f PRIVATE project_options)
target_link_libraries(backends.cuda.image16f PUBLIC  
						backends.cuda.image.arithmetic16f
						backends.cuda.image.dataExchangeAndInit16f 
						backends.cuda.image.filtering16f 
						backends.cuda.image.geometryTransforms16f 
						backends.cuda.image.morphology16f 
						backends.cuda.image.statistics16f 
						backends.cuda.image.thresholdAndCompare16f 
						backends.cuda.image.filteringBilateral 
						backends.cuda.image.dataExchangeAndInitTranspose
						backends.cuda
						common
						)



#32f
add_library(backends.cuda.image32f SHARED
						imageView32f.cpp
						)
						
target_compile_definitions(backends.cuda.image32f PUBLIC -DMPP_CUDAI) 
target_include_directories(backends.cuda.image32f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image32f PRIVATE project_options)
target_link_libraries(backends.cuda.image32f PUBLIC  
						backends.cuda.image.arithmetic32f
						backends.cuda.image.dataExchangeAndInit32f 
						backends.cuda.image.filtering32f 
						backends.cuda.image.geometryTransforms32f 
						backends.cuda.image.morphology32f 
						backends.cuda.image.statistics32f 
						backends.cuda.image.thresholdAndCompare32f 
						backends.cuda.image.filteringBilateral 
						backends.cuda.image.dataExchangeAndInitTranspose
						backends.cuda
						common
						)



#64f
add_library(backends.cuda.image64f SHARED
						imageView64f.cpp
						)
						
target_compile_definitions(backends.cuda.image64f PUBLIC -DMPP_CUDAI)  
target_include_directories(backends.cuda.image64f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image64f PRIVATE project_options)
target_link_libraries(backends.cuda.image64f PUBLIC  
						backends.cuda.image.arithmetic64f
						backends.cuda.image.dataExchangeAndInit64f 
						backends.cuda.image.filtering64f 
						backends.cuda.image.geometryTransforms64f 
						backends.cuda.image.morphology64f 
						backends.cuda.image.statistics64f 
						backends.cuda.image.thresholdAndCompare64f 
						backends.cuda.image.dataExchangeAndInitTranspose
						backends.cuda
						common
						)
						
