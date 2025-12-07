add_library(backends.cuda.image.geometryTransforms8s OBJECT
	8s/affineBack.cu
	8s/mirror.cu
	8s/perspectiveBack.cu
	8s/remap.cu
	8s/resize.cu)

add_library(backends.cuda.image.geometryTransforms8u OBJECT
	8u/affineBack.cu
	8u/mirror.cu
	8u/perspectiveBack.cu
	8u/remap.cu
	8u/resize.cu)

add_library(backends.cuda.image.geometryTransforms16s OBJECT
	16s/affineBack.cu
	16s/mirror.cu
	16s/perspectiveBack.cu
	16s/remap.cu
	16s/resize.cu)

add_library(backends.cuda.image.geometryTransforms16u OBJECT
	16u/affineBack.cu
	16u/mirror.cu
	16u/perspectiveBack.cu
	16u/remap.cu
	16u/resize.cu)

add_library(backends.cuda.image.geometryTransforms32s OBJECT
	32s/affineBack.cu
	32s/mirror.cu
	32s/perspectiveBack.cu
	32s/remap.cu
	32s/resize.cu)

add_library(backends.cuda.image.geometryTransforms32u OBJECT
	32u/affineBack.cu
	32u/mirror.cu
	32u/perspectiveBack.cu
	32u/remap.cu
	32u/resize.cu)

add_library(backends.cuda.image.geometryTransforms16f OBJECT
	16f/affineBack.cu
	16f/mirror.cu
	16f/perspectiveBack.cu
	16f/remap.cu
	16f/resize.cu)

add_library(backends.cuda.image.geometryTransforms16bf OBJECT
	16bf/affineBack.cu
	16bf/mirror.cu
	16bf/perspectiveBack.cu
	16bf/remap.cu
	16bf/resize.cu)

add_library(backends.cuda.image.geometryTransforms32f OBJECT
	32f/affineBack.cu
	32f/mirror.cu
	32f/perspectiveBack.cu
	32f/remap.cu
	32f/resize.cu)

add_library(backends.cuda.image.geometryTransforms64f OBJECT
	64f/affineBack.cu
	64f/mirror.cu
	64f/perspectiveBack.cu
	64f/remap.cu
	64f/resize.cu)

add_library(backends.cuda.image.geometryTransforms16sc OBJECT
	16sc/affineBack.cu
	16sc/mirror.cu
	16sc/perspectiveBack.cu
	16sc/remap.cu
	16sc/resize.cu)

add_library(backends.cuda.image.geometryTransforms32sc OBJECT
	32sc/affineBack.cu
	32sc/mirror.cu
	32sc/perspectiveBack.cu
	32sc/remap.cu
	32sc/resize.cu)

add_library(backends.cuda.image.geometryTransforms32fc OBJECT
	32fc/affineBack.cu	
	32fc/mirror.cu	
	32fc/perspectiveBack.cu	
	32fc/remap.cu
	32fc/resize.cu
	)
	
	
	
target_include_directories(backends.cuda.image.geometryTransforms8s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms8s PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms8u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms8u PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms16s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms16s PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms16u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms16u PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms32s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms32s PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms32u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms32u PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms16sc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms16sc PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms32sc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms32sc PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms16bf PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms16bf PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms16f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms16f PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms32f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms32f PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms32fc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms32fc PRIVATE project_options)

target_include_directories(backends.cuda.image.geometryTransforms64f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(backends.cuda.image.geometryTransforms64f PRIVATE project_options)