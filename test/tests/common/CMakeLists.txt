add_library(common_test_halffp16_cuda_kernel test_halffp16_kernel.cu)
target_link_libraries(common_test_halffp16_cuda_kernel PRIVATE project_options)
target_include_directories(common_test_halffp16_cuda_kernel  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )

add_library(common_test_bfloat16_cuda_kernel test_bfloat16_kernel.cu)
target_link_libraries(common_test_bfloat16_cuda_kernel PRIVATE project_options)
target_include_directories(common_test_bfloat16_cuda_kernel  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
						  
add_library(common_test_typecast_cuda_kernel test_typecast_kernel.cu)
target_link_libraries(common_test_typecast_cuda_kernel PRIVATE project_options)
target_include_directories(common_test_typecast_cuda_kernel  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )

add_library(common_test_vector1_cuda_kernel test_vector1_kernel.cu)
target_link_libraries(common_test_vector1_cuda_kernel PRIVATE project_options)
target_include_directories(common_test_vector1_cuda_kernel  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )

add_library(common_test_vector2_cuda_kernel test_vector2_kernel.cu)
target_link_libraries(common_test_vector2_cuda_kernel PRIVATE project_options)
target_include_directories(common_test_vector2_cuda_kernel  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )

add_library(common_test_vector3_cuda_kernel test_vector3_kernel.cu)
target_link_libraries(common_test_vector3_cuda_kernel PRIVATE project_options)
target_include_directories(common_test_vector3_cuda_kernel  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )

add_library(common_test_vector4_cuda_kernel test_vector4_kernel.cu)
target_link_libraries(common_test_vector4_cuda_kernel PRIVATE project_options)
target_include_directories(common_test_vector4_cuda_kernel  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )

add_library(common_test_vector4A_cuda_kernel test_vector4A_kernel.cu)
target_link_libraries(common_test_vector4A_cuda_kernel PRIVATE project_options)
target_include_directories(common_test_vector4A_cuda_kernel  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )

addCatchTest(test_bfloat16.cpp "common" "BFloat16")
addCatchTest(test_bfloat16_cuda.cpp "common;common_test_bfloat16_cuda_kernel;backends.cuda" "BFloat16Cuda")
addCatchTest(test_complex.cpp "common" "Complex")
addCatchTest(test_halffp16.cpp "common" "HalfFP16")
addCatchTest(test_halffp16_cuda.cpp "common;common_test_halffp16_cuda_kernel;backends.cuda" "HalfFP16Cuda")
addCatchTest(test_safeCast.cpp "common" "SafeCast")
addCatchTest(test_scratchBuffer.cpp "common" "ScratchBuffer")
addCatchTest(test_tupel.cpp "common" "Tupel")
addCatchTest(test_utilities.cpp "" "Utilities")
addCatchTest(test_typecast.cpp "common" "Typecast")
addCatchTest(test_typecast_cuda.cpp "common;common_test_typecast_cuda_kernel;backends.cuda" "TypecastCuda")
addCatchTest(test_vector1.cpp "common" "Vector1")
addCatchTest(test_vector1_cuda.cpp "common;common_test_vector1_cuda_kernel;backends.cuda" "Vector1Cuda")
addCatchTest(test_vector2.cpp "common" "Vector2")
addCatchTest(test_vector2_cuda.cpp "common;common_test_vector2_cuda_kernel;backends.cuda" "Vector2Cuda")
addCatchTest(test_vector3.cpp "common" "Vector3")
addCatchTest(test_vector3_cuda.cpp "common;common_test_vector3_cuda_kernel;backends.cuda" "Vector3Cuda")
addCatchTest(test_vector4.cpp "common" "Vector4")
addCatchTest(test_vector4_cuda.cpp "common;common_test_vector4_cuda_kernel;backends.cuda" "Vector4Cuda")
addCatchTest(test_vector4A.cpp "common" "Vector4A")
addCatchTest(test_vector4A_cuda.cpp "common;common_test_vector4A_cuda_kernel;backends.cuda" "Vector4ACuda")

add_subdirectory(image)
