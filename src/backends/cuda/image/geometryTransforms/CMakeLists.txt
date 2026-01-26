add_library(backends.cuda.image.geometryTransforms8s OBJECT
	8s/affineBack.cu
	8s/affineBack_p2.cu
	8s/affineBack_p3.cu
	8s/affineBack_p4.cu
	8s/mirror.cu
	8s/perspectiveBack.cu
	8s/perspectiveBack_p2.cu
	8s/perspectiveBack_p3.cu
	8s/perspectiveBack_p4.cu
	8s/remap.cu
	8s/remap_p2.cu
	8s/remap_p3.cu
	8s/remap_p4.cu
	8s/remap2.cu
	8s/remap2_p2.cu
	8s/remap2_p3.cu
	8s/remap2_p4.cu
	8s/resize.cu
	8s/resize_p2.cu
	8s/resize_p3.cu
	8s/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms8u OBJECT
	8u/affineBack.cu
	8u/affineBack_p2.cu
	8u/affineBack_p3.cu
	8u/affineBack_p4.cu
	8u/mirror.cu
	8u/perspectiveBack.cu
	8u/perspectiveBack_p2.cu
	8u/perspectiveBack_p3.cu
	8u/perspectiveBack_p4.cu
	8u/remap.cu
	8u/remap_p2.cu
	8u/remap_p3.cu
	8u/remap_p4.cu
	8u/remap2.cu
	8u/remap2_p2.cu
	8u/remap2_p3.cu
	8u/remap2_p4.cu
	8u/resize.cu
	8u/resize_p2.cu
	8u/resize_p3.cu
	8u/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms16s OBJECT
	16s/affineBack.cu
	16s/affineBack_p2.cu
	16s/affineBack_p3.cu
	16s/affineBack_p4.cu
	16s/mirror.cu
	16s/perspectiveBack.cu
	16s/perspectiveBack_p2.cu
	16s/perspectiveBack_p3.cu
	16s/perspectiveBack_p4.cu
	16s/remap.cu
	16s/remap_p2.cu
	16s/remap_p3.cu
	16s/remap_p4.cu
	16s/remap2.cu
	16s/remap2_p2.cu
	16s/remap2_p3.cu
	16s/remap2_p4.cu
	16s/resize.cu
	16s/resize_p2.cu
	16s/resize_p3.cu
	16s/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms16u OBJECT
	16u/affineBack.cu
	16u/affineBack_p2.cu
	16u/affineBack_p3.cu
	16u/affineBack_p4.cu
	16u/mirror.cu
	16u/perspectiveBack.cu
	16u/perspectiveBack_p2.cu
	16u/perspectiveBack_p3.cu
	16u/perspectiveBack_p4.cu
	16u/remap.cu
	16u/remap_p2.cu
	16u/remap_p3.cu
	16u/remap_p4.cu
	16u/remap2.cu
	16u/remap2_p2.cu
	16u/remap2_p3.cu
	16u/remap2_p4.cu
	16u/resize.cu
	16u/resize_p2.cu
	16u/resize_p3.cu
	16u/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms32s OBJECT
	32s/affineBack.cu
	32s/affineBack_p2.cu
	32s/affineBack_p3.cu
	32s/affineBack_p4.cu
	32s/mirror.cu
	32s/perspectiveBack.cu
	32s/perspectiveBack_p2.cu
	32s/perspectiveBack_p3.cu
	32s/perspectiveBack_p4.cu
	32s/remap.cu
	32s/remap_p2.cu
	32s/remap_p3.cu
	32s/remap_p4.cu
	32s/remap2.cu
	32s/remap2_p2.cu
	32s/remap2_p3.cu
	32s/remap2_p4.cu
	32s/resize.cu
	32s/resize_p2.cu
	32s/resize_p3.cu
	32s/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms32u OBJECT
	32u/affineBack.cu
	32u/affineBack_p2.cu
	32u/affineBack_p3.cu
	32u/affineBack_p4.cu
	32u/mirror.cu
	32u/perspectiveBack.cu
	32u/perspectiveBack_p2.cu
	32u/perspectiveBack_p3.cu
	32u/perspectiveBack_p4.cu
	32u/remap.cu
	32u/remap_p2.cu
	32u/remap_p3.cu
	32u/remap_p4.cu
	32u/remap2.cu
	32u/remap2_p2.cu
	32u/remap2_p3.cu
	32u/remap2_p4.cu
	32u/resize.cu
	32u/resize_p2.cu
	32u/resize_p3.cu
	32u/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms16f OBJECT
	16f/affineBack.cu
	16f/affineBack_p2.cu
	16f/affineBack_p3.cu
	16f/affineBack_p4.cu
	16f/mirror.cu
	16f/perspectiveBack.cu
	16f/perspectiveBack_p2.cu
	16f/perspectiveBack_p3.cu
	16f/perspectiveBack_p4.cu
	16f/remap.cu
	16f/remap_p2.cu
	16f/remap_p3.cu
	16f/remap_p4.cu
	16f/remap2.cu
	16f/remap2_p2.cu
	16f/remap2_p3.cu
	16f/remap2_p4.cu
	16f/resize.cu
	16f/resize_p2.cu
	16f/resize_p3.cu
	16f/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms16bf OBJECT
	16bf/affineBack.cu
	16bf/affineBack_p2.cu
	16bf/affineBack_p3.cu
	16bf/affineBack_p4.cu
	16bf/mirror.cu
	16bf/perspectiveBack.cu
	16bf/perspectiveBack_p2.cu
	16bf/perspectiveBack_p3.cu
	16bf/perspectiveBack_p4.cu
	16bf/remap.cu
	16bf/remap_p2.cu
	16bf/remap_p3.cu
	16bf/remap_p4.cu
	16bf/remap2.cu
	16bf/remap2_p2.cu
	16bf/remap2_p3.cu
	16bf/remap2_p4.cu
	16bf/resize.cu
	16bf/resize_p2.cu
	16bf/resize_p3.cu
	16bf/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms32f OBJECT
	32f/affineBack.cu
	32f/affineBack_p2.cu
	32f/affineBack_p3.cu
	32f/affineBack_p4.cu
	32f/mirror.cu
	32f/perspectiveBack.cu
	32f/perspectiveBack_p2.cu
	32f/perspectiveBack_p3.cu
	32f/perspectiveBack_p4.cu
	32f/remap.cu
	32f/remap_p2.cu
	32f/remap_p3.cu
	32f/remap_p4.cu
	32f/remap2.cu
	32f/remap2_p2.cu
	32f/remap2_p3.cu
	32f/remap2_p4.cu
	32f/resize.cu
	32f/resize_p2.cu
	32f/resize_p3.cu
	32f/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms64f OBJECT
	64f/affineBack.cu
	64f/affineBack_p2.cu
	64f/affineBack_p3.cu
	64f/affineBack_p4.cu
	64f/mirror.cu
	64f/perspectiveBack.cu
	64f/perspectiveBack_p2.cu
	64f/perspectiveBack_p3.cu
	64f/perspectiveBack_p4.cu
	64f/remap.cu
	64f/remap_p2.cu
	64f/remap_p3.cu
	64f/remap_p4.cu
	64f/remap2.cu
	64f/remap2_p2.cu
	64f/remap2_p3.cu
	64f/remap2_p4.cu
	64f/resize.cu
	64f/resize_p2.cu
	64f/resize_p3.cu
	64f/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms16sc OBJECT
	16sc/affineBack.cu
	16sc/affineBack_p2.cu
	16sc/affineBack_p3.cu
	16sc/affineBack_p4.cu
	16sc/mirror.cu
	16sc/perspectiveBack.cu
	16sc/perspectiveBack_p2.cu
	16sc/perspectiveBack_p3.cu
	16sc/perspectiveBack_p4.cu
	16sc/remap.cu
	16sc/remap_p2.cu
	16sc/remap_p3.cu
	16sc/remap_p4.cu
	16sc/remap2.cu
	16sc/remap2_p2.cu
	16sc/remap2_p3.cu
	16sc/remap2_p4.cu
	16sc/resize.cu
	16sc/resize_p2.cu
	16sc/resize_p3.cu
	16sc/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms32sc OBJECT
	32sc/affineBack.cu
	32sc/affineBack_p2.cu
	32sc/affineBack_p3.cu
	32sc/affineBack_p4.cu
	32sc/mirror.cu
	32sc/perspectiveBack.cu
	32sc/perspectiveBack_p2.cu
	32sc/perspectiveBack_p3.cu
	32sc/perspectiveBack_p4.cu
	32sc/remap.cu
	32sc/remap_p2.cu
	32sc/remap_p3.cu
	32sc/remap_p4.cu
	32sc/remap2.cu
	32sc/remap2_p2.cu
	32sc/remap2_p3.cu
	32sc/remap2_p4.cu
	32sc/resize.cu
	32sc/resize_p2.cu
	32sc/resize_p3.cu
	32sc/resize_p4.cu)

add_library(backends.cuda.image.geometryTransforms32fc OBJECT
	32fc/affineBack.cu
	32fc/affineBack_p2.cu
	32fc/affineBack_p3.cu
	32fc/affineBack_p4.cu
	32fc/mirror.cu
	32fc/perspectiveBack.cu
	32fc/perspectiveBack_p2.cu
	32fc/perspectiveBack_p3.cu
	32fc/perspectiveBack_p4.cu
	32fc/remap.cu
	32fc/remap_p2.cu
	32fc/remap_p3.cu
	32fc/remap_p4.cu
	32fc/remap2.cu
	32fc/remap2_p2.cu
	32fc/remap2_p3.cu 
	32fc/remap2_p4.cu
	32fc/resize.cu
	32fc/resize_p2.cu
	32fc/resize_p3.cu
	32fc/resize_p4.cu)
	
	
	
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