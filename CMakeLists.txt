cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()

project(OPP VERSION 0.1.0 LANGUAGES CUDA CXX)

LIST( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake/modules
)

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)
# Link this 'library' to set the c++ standard / compile-time options requested

add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)
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

set(COPYRIGHT_MESSAGE "(c) 2024 Michael Kunz")
configure_file(${PROJECT_SOURCE_DIR}/src/common/version.h.in ${PROJECT_SOURCE_DIR}/src/common/version.h @ONLY)

# include folder for external dependencies:
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/external/include)
# include main src folder:
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/src)

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





