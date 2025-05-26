cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep -g -G")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()

if (WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /EHsc")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/bigobj -Xcompiler=/EHsc")
elseif (UNIX)
endif()

project(OPP VERSION 0.1.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

LIST( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake/modules
)

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)
# Link this 'library' to set the c++ standard / compile-time options requested

add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)
target_compile_features(project_options INTERFACE cuda_std_20)
target_compile_definitions(project_options INTERFACE -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")  
if (APPLE)
	set(CMAKE_CXX_VISIBILITY_PRESET hidden)
	set_target_properties(project_options PROPERTIES CXX_VISIBILITY_PRESET hidden)
elseif (UNIX)
	#set(CMAKE_CXX_VISIBILITY_PRESET hidden)
	#set_target_properties(project_options PROPERTIES CXX_VISIBILITY_PRESET hidden)
    target_link_libraries(project_options INTERFACE tbb)
endif()

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options if supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

#we need version >18 of LLVM
if (WIN32)
	set(LLVM_DIR "C:/Program Files/LLVM18")
elseif (UNIX)
	set(LLVM_DIR "/usr/lib/llvm-18/")
endif()

set(CPPCHECK_HINT_DIR "")
set(CLANGTIDY_HINT_DIR "${LLVM_DIR}/bin")
set(INCLUDE_WHAT_YOU_USE_HINT_DIR "")

# allow for static analysis options
include(cmake/StaticAnalyzers.cmake)

option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)
option(ENABLE_TESTING "Enable Test Builds" ON)
set(INSTALLDEST "${CMAKE_SOURCE_DIR}/install")

set(COPYRIGHT_MESSAGE "(c) 2025 Michael Kunz")
configure_file(${PROJECT_SOURCE_DIR}/src/common/version.h.in ${PROJECT_SOURCE_DIR}/src/common/version.h @ONLY)

# include folder for external dependencies:
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/external/include)
# include main src folder:
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/src)

#################################################################
########## Enable or disable backends and pixel types ###########
#################################################################

# we need some cuda core componentes for CUDA and NPP backend:
option(OPP_ENABLE_CUDA_CORE "cuda core components needed for CUDA and NPP backend" ON)
option(OPP_ENABLE_CUDA_BACKEND "compile and build OPP with the backend for CUDA platforms" ON)
option(OPP_ENABLE_NPP_BACKEND "compile and build OPP with the NPP wrapper" ON)
option(OPP_ENABLE_HIP_BACKEND "compile and build OPP with the backend for HIP platforms" false) 
option(OPP_ENABLE_CPUSIMPLE_BACKEND "compile and build OPP with the backend with a simple CPU implementation (for testing and validation)" ON) 


option(OPPi_ENABLE_DOUBLE_TYPE "enable support for double type images" ON)
option(OPPi_ENABLE_FLOAT_TYPE "enable support for float type images" ON)
option(OPPi_ENABLE_HALFFLOAT16_TYPE "enable support for half-float16 type images" ON)
option(OPPi_ENABLE_BFLOAT16_TYPE "enable support for BFloat16 type images" ON)
option(OPPi_ENABLE_INT64_TYPE "enable support for int64 type images" ON)
option(OPPi_ENABLE_UINT64_TYPE "enable support for uint64 type images" ON)
option(OPPi_ENABLE_INT32_TYPE "enable support for int32 type images" ON)
option(OPPi_ENABLE_UINT32_TYPE "enable support for uint32 type images" ON)
option(OPPi_ENABLE_INT16_TYPE "enable support for int16 type images" ON)
option(OPPi_ENABLE_UINT16_TYPE "enable support for uint16 type images" ON)
option(OPPi_ENABLE_INT8_TYPE "enable support for int8 type images" ON)
option(OPPi_ENABLE_UINT8_TYPE "enable support for uint8 type images" ON)

option(OPPi_ENABLE_COMPLEX_DOUBLE_TYPE "enable support for complex double type images" OFF)
option(OPPi_ENABLE_COMPLEX_FLOAT_TYPE "enable support for complex float type images" ON)
option(OPPi_ENABLE_COMPLEX_HALFFLOAT16_TYPE "enable support for complex half-float16 type images" OFF)
option(OPPi_ENABLE_COMPLEX_BFLOAT16_TYPE "enable support for complex BFloat16 type images" OFF)
option(OPPi_ENABLE_COMPLEX_INT64_TYPE "enable support for complex int32 type images" ON)
option(OPPi_ENABLE_COMPLEX_INT32_TYPE "enable support for complex int32 type images" ON)
option(OPPi_ENABLE_COMPLEX_INT16_TYPE "enable support for complex int16 type images" ON)

option(OPPi_ENABLE_ONE_CHANNEL "enable support for one channel images" ON)
option(OPPi_ENABLE_TWO_CHANNEL "enable support for two channel images" ON)
option(OPPi_ENABLE_THREE_CHANNEL "enable support for three channel images" ON)
option(OPPi_ENABLE_FOUR_CHANNEL "enable support for four channel images" ON)
option(OPPi_ENABLE_FOURALPHA_CHANNEL "enable support for four channel images with ignored alpha" ON)

# as cmake accepts nearly everything for bool, manually set the C-precompiler definition to either 0 or 1
if (OPP_ENABLE_CUDA_CORE)
target_compile_definitions(project_options INTERFACE -DOPP_ENABLE_CUDA_CORE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPP_ENABLE_CUDA_CORE=0)
endif()

if (OPP_ENABLE_CUDA_BACKEND)
target_compile_definitions(project_options INTERFACE -DOPP_ENABLE_CUDA_BACKEND=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPP_ENABLE_CUDA_BACKEND=0)
endif()

if (OPP_ENABLE_NPP_BACKEND)
target_compile_definitions(project_options INTERFACE -DOPP_ENABLE_NPP_BACKEND=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPP_ENABLE_NPP_BACKEND=0)
endif()

if (OPP_ENABLE_HIP_BACKEND)
target_compile_definitions(project_options INTERFACE -DOPP_ENABLE_HIP_BACKEND=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPP_ENABLE_HIP_BACKEND=0)
endif()

if (OPP_ENABLE_CPUSIMPLE_BACKEND)
target_compile_definitions(project_options INTERFACE -DOPP_ENABLE_CPUSIMPLE_BACKEND=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPP_ENABLE_CPUSIMPLE_BACKEND=0)
endif()



if (OPPi_ENABLE_DOUBLE_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_DOUBLE_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_DOUBLE_TYPE=0)
endif()
if (OPPi_ENABLE_FLOAT_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_FLOAT_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_FLOAT_TYPE=0)
endif()
if (OPPi_ENABLE_HALFFLOAT16_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_HALFFLOAT16_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_HALFFLOAT16_TYPE=0)
endif()
if (OPPi_ENABLE_BFLOAT16_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_BFLOAT16_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_BFLOAT16_TYPE=0)
endif()
if (OPPi_ENABLE_INT64_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_INT64_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_INT64_TYPE=0)
endif()
if (OPPi_ENABLE_UINT64_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_UINT64_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_UINT64_TYPE=0)
endif()
if (OPPi_ENABLE_INT32_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_INT32_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_INT32_TYPE=0)
endif()
if (OPPi_ENABLE_UINT32_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_UINT32_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_UINT32_TYPE=0)
endif()
if (OPPi_ENABLE_INT16_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_INT16_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_INT16_TYPE=0)
endif()
if (OPPi_ENABLE_UINT16_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_UINT16_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_UINT16_TYPE=0)
endif()
if (OPPi_ENABLE_INT8_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_INT8_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_INT8_TYPE=0)
endif()
if (OPPi_ENABLE_UINT8_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_UINT8_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_UINT8_TYPE=0)
endif()

if (OPPi_ENABLE_COMPLEX_DOUBLE_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_DOUBLE_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_DOUBLE_TYPE=0)
endif()
if (OPPi_ENABLE_COMPLEX_FLOAT_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_FLOAT_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_FLOAT_TYPE=0)
endif()
if (OPPi_ENABLE_COMPLEX_HALFFLOAT16_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_HALFFLOAT16_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_HALFFLOAT16_TYPE=0)
endif()
if (OPPi_ENABLE_COMPLEX_BFLOAT16_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_BFLOAT16_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_BFLOAT16_TYPE=0)
endif()
if (OPPi_ENABLE_COMPLEX_INT64_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_INT64_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_INT64_TYPE=0)
endif()
if (OPPi_ENABLE_COMPLEX_INT32_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_INT32_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_INT32_TYPE=0)
endif()
if (OPPi_ENABLE_COMPLEX_INT16_TYPE)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_INT16_TYPE=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_COMPLEX_INT16_TYPE=0)
endif()

if (OPPi_ENABLE_ONE_CHANNEL)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_ONE_CHANNEL=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_ONE_CHANNEL=0)
endif()
if (OPPi_ENABLE_TWO_CHANNEL)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_TWO_CHANNEL=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_TWO_CHANNEL=0)
endif()
if (OPPi_ENABLE_THREE_CHANNEL)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_THREE_CHANNEL=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_THREE_CHANNEL=0)
endif()
if (OPPi_ENABLE_FOUR_CHANNEL)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_FOUR_CHANNEL=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_FOUR_CHANNEL=0)
endif()
if (OPPi_ENABLE_FOURALPHA_CHANNEL)
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_FOURALPHA_CHANNEL=1)  
else()
target_compile_definitions(project_options INTERFACE -DOPPi_ENABLE_FOURALPHA_CHANNEL=0)
endif()
#################################################################




######################### System ################################
if(WIN32)
elseif(APPLE)
else()
	#set(THREADS_PREFER_PTHREAD_FLAG ON)
	#find_package(Threads REQUIRED)
endif()
######################### System ################################

######################### LIBCLANG ##############################

find_package(LIBCLANG REQUIRED)

######################### LIBPNG ################################

######################### CUDA ##################################

if (NOT APPLE)
	find_package(CUDAToolkit 12.0 REQUIRED)
	# we need it in order to link to NPP
endif()
#find_package(CUDAToolkit REQUIRED)

######################### CUDA ##################################


########################## ZLIB ##########################
if(WIN32)
	add_library(zlib STATIC IMPORTED)
	set_target_properties(zlib PROPERTIES
		IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/external/lib/zlib/zlibstatic.lib
		INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/external/include/"
	)
	link_directories(${CMAKE_SOURCE_DIR}/external/lib/zlib/)
elseif(APPLE)
	find_package( ZLIB REQUIRED )
else()
	add_library(zlib STATIC IMPORTED)
	set_target_properties(zlib PROPERTIES
		IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/external/lib/zlib/libz.a
		INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/external/include/"
	)
	link_directories(${CMAKE_SOURCE_DIR}/external/lib/zlib/)
endif()
##########################################################

######################### LIBPNG ##############################

#if(WIN32)
#	# On windows, use the lib compiled by us:
#	add_library(PNG::PNG SHARED IMPORTED)
#	set_target_properties(PNG::PNG PROPERTIES
#		IMPORTED_LOCATION "F:/ImportantLibs/lpng1643/projects/vstudio/x64/Debug/libpng16.dll"		
#		IMPORTED_IMPLIB "F:/ImportantLibs/lpng1643/projects/vstudio/x64/Debug/libpng16.lib"
#		INTERFACE_INCLUDE_DIRECTORIES "F:/ImportantLibs/lpng1643/"
#	)
#	#link_directories("F:/ImportantLibs/libpng/projects/vstudio/Debug/")
#
#	set(PNG_FOUND TRUE)
#elseif(APPLE)
#	# use libpng from system:
#	find_package(PNG REQUIRED)
#else()
#	# use libpng from system:
#	find_package(PNG REQUIRED)
#endif()

######################### LIBPNG ################################


add_subdirectory(src)

if(ENABLE_TESTING)
  set(CMAKE_CXX_CLANG_TIDY "") #no clang-tidy in tests
  include(cmake/Catch.cmake)
  add_subdirectory(test/catch2)

  enable_testing()
  message("Building Tests.")
  add_subdirectory(test)
endif()





