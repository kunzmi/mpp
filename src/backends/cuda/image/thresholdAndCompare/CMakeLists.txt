add_library(backends.cuda.image.thresholdAndCompare8s OBJECT
	8s/compare.cu
	8s/replaceIf.cu
	8s/threshold.cu)
	
add_library(backends.cuda.image.thresholdAndCompare8u OBJECT
	8u/compare.cu
	8u/replaceIf.cu
	8u/threshold.cu)
	
add_library(backends.cuda.image.thresholdAndCompare16s OBJECT
	16s/compare.cu
	16s/replaceIf.cu
	16s/threshold.cu)
	
add_library(backends.cuda.image.thresholdAndCompare16u OBJECT
	16u/compare.cu
	16u/replaceIf.cu
	16u/threshold.cu)
	
add_library(backends.cuda.image.thresholdAndCompare16f OBJECT
	16f/compare.cu
	16f/compareEqEps.cu
	16f/replaceIf.cu
	16f/threshold.cu)
	
add_library(backends.cuda.image.thresholdAndCompare16bf OBJECT
	16bf/compare.cu
	16bf/compareEqEps.cu
	16bf/replaceIf.cu
	16bf/threshold.cu)
	
add_library(backends.cuda.image.thresholdAndCompare32s OBJECT
	32s/compare.cu
	32s/replaceIf.cu
	32s/threshold.cu)
	
add_library(backends.cuda.image.thresholdAndCompare32u OBJECT
	32u/compare.cu
	32u/replaceIf.cu
	32u/threshold.cu)
	
add_library(backends.cuda.image.thresholdAndCompare32f OBJECT
	32f/compare.cu
	32f/compareEqEps.cu
	32f/replaceIf.cu
	32f/threshold.cu)
	
add_library(backends.cuda.image.thresholdAndCompare64f OBJECT
	64f/compare.cu
	64f/compareEqEps.cu
	64f/replaceIf.cu
	64f/threshold.cu)
	
add_library(backends.cuda.image.thresholdAndCompare16sc OBJECT
	16sc/compare.cu
	16sc/replaceIf.cu)
	
add_library(backends.cuda.image.thresholdAndCompare32sc OBJECT
	32sc/compare.cu
	32sc/replaceIf.cu)
	
add_library(backends.cuda.image.thresholdAndCompare32fc OBJECT
	32fc/compare.cu
	32fc/compareEqEps.cu
	32fc/replaceIf.cu
	)
	
target_include_directories(backends.cuda.image.thresholdAndCompare8s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare8s PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare8u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare8u PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare16s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare16s PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare16u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare16u PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare32s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare32s PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare32u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare32u PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare16sc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare16sc PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare32sc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare32sc PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare16bf PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare16bf PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare16f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare16f PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare32f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare32f PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare32fc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare32fc PRIVATE project_options)

target_include_directories(backends.cuda.image.thresholdAndCompare64f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.thresholdAndCompare64f PRIVATE project_options)