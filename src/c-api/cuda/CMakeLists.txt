
add_library(mppci_capi_8u SHARED
	mppci_capi_8u.cpp
	convertScale_8u.cpp
	copySwapChannelDup_8u.cpp ) 
target_link_libraries(mppci_capi_8u PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_8u PUBLIC  						
						backends.cuda.image.arithmetic8u
						backends.cuda.image.colorConversion8u
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
target_link_options(mppci_capi_8u PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_8u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_8u PROPERTIES OUTPUT_NAME mppci_capi_8u)
					

add_library(mppci_capi_8s SHARED
	mppci_capi_8s.cpp
	convertScale_8s.cpp
	copySwapChannelDup_8s.cpp ) 
target_link_libraries(mppci_capi_8s PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_8s PUBLIC  					
						backends.cuda.image.arithmetic8s
						backends.cuda.image.colorConversion8s
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
target_link_options(mppci_capi_8s PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_8s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_8s PROPERTIES OUTPUT_NAME mppci_capi_8s)
					

add_library(mppci_capi_16u SHARED
	mppci_capi_16u.cpp
	convertScale_16u.cpp
	copySwapChannelDup_16u.cpp ) 
target_link_libraries(mppci_capi_16u PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_16u PUBLIC  			
						backends.cuda.image.arithmetic16u
						backends.cuda.image.colorConversion16u
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
target_link_options(mppci_capi_16u PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_16u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_16u PROPERTIES OUTPUT_NAME mppci_capi_16u)
					

add_library(mppci_capi_16s SHARED
	mppci_capi_16s.cpp
	convertScale_16s.cpp
	copySwapChannelDup_16s.cpp ) 
target_link_libraries(mppci_capi_16s PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_16s PUBLIC  			
						backends.cuda.image.arithmetic16s
						backends.cuda.image.colorConversion16s
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
target_link_options(mppci_capi_16s PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_16s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_16s PROPERTIES OUTPUT_NAME mppci_capi_16s)
					

add_library(mppci_capi_32u SHARED
	mppci_capi_32u.cpp
	convertScale_32u.cpp
	copySwapChannelDup_32u.cpp ) 
target_link_libraries(mppci_capi_32u PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_32u PUBLIC  		
						backends.cuda.image.arithmetic32u
						backends.cuda.image.colorConversion32u
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
target_link_options(mppci_capi_32u PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_32u PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_32u PROPERTIES OUTPUT_NAME mppci_capi_32u)
					

add_library(mppci_capi_32s SHARED
	mppci_capi_32s.cpp
	convertScale_32s.cpp
	copySwapChannelDup_32s.cpp ) 
target_link_libraries(mppci_capi_32s PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_32s PUBLIC  				
						backends.cuda.image.arithmetic32s
						backends.cuda.image.colorConversion32s
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
target_link_options(mppci_capi_32s PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_32s PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_32s PROPERTIES OUTPUT_NAME mppci_capi_32s)
					

add_library(mppci_capi_16f SHARED
	mppci_capi_16f.cpp
	convertScale_16f.cpp
	copySwapChannelDup_16f.cpp ) 
target_link_libraries(mppci_capi_16f PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_16f PUBLIC  	
						backends.cuda.image.arithmetic16f
						backends.cuda.image.colorConversion16f
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
target_link_options(mppci_capi_16f PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_16f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_16f PROPERTIES OUTPUT_NAME mppci_capi_16f)
					

add_library(mppci_capi_16bf SHARED
	mppci_capi_16bf.cpp
	convertScale_16bf.cpp
	copySwapChannelDup_16bf.cpp ) 
target_link_libraries(mppci_capi_16bf PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_16bf PUBLIC  		
						backends.cuda.image.arithmetic16bf
						backends.cuda.image.colorConversion16bf
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
target_link_options(mppci_capi_16bf PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_16bf PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_16bf PROPERTIES OUTPUT_NAME mppci_capi_16bf)
					

add_library(mppci_capi_32f SHARED
	mppci_capi_32f.cpp
	convertScale_32f.cpp
	copySwapChannelDup_32f.cpp ) 
target_link_libraries(mppci_capi_32f PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_32f PUBLIC  	 
						backends.cuda.image.arithmetic32f
						backends.cuda.image.colorConversion32f
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
target_link_options(mppci_capi_32f PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_32f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_32f PROPERTIES OUTPUT_NAME mppci_capi_32f)
					

add_library(mppci_capi_64f SHARED
	mppci_capi_64f.cpp
	convertScale_64f.cpp
	copySwapChannelDup_64f.cpp ) 
target_link_libraries(mppci_capi_64f PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_64f PUBLIC  
						backends.cuda.image.arithmetic64f
						backends.cuda.image.colorConversion64f
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
target_link_options(mppci_capi_64f PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_64f PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_64f PROPERTIES OUTPUT_NAME mppci_capi_64f)
					

add_library(mppci_capi_16sc SHARED
	mppci_capi_16sc.cpp
	convertScale_16sc.cpp
	copySwapChannelDup_16sc.cpp ) 
target_link_libraries(mppci_capi_16sc PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_16sc PUBLIC  
						backends.cuda.image.arithmetic16sc
						backends.cuda.image.dataExchangeAndInit16sc 
						backends.cuda.image.filtering16sc 
						backends.cuda.image.geometryTransforms16sc 
						backends.cuda.image.statistics16sc
						backends.cuda.image.thresholdAndCompare16sc 
						backends.cuda
						common
						)
target_link_options(mppci_capi_16sc PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_16sc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_16sc PROPERTIES OUTPUT_NAME mppci_capi_16sc)
					

add_library(mppci_capi_32sc SHARED
	mppci_capi_32sc.cpp
	convertScale_32sc.cpp
	copySwapChannelDup_32sc.cpp ) 
target_link_libraries(mppci_capi_32sc PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_32sc PUBLIC  	
						backends.cuda.image.arithmetic32sc
						backends.cuda.image.dataExchangeAndInit32sc 
						backends.cuda.image.filtering32sc 
						backends.cuda.image.geometryTransforms32sc 
						backends.cuda.image.statistics32sc 
						backends.cuda.image.thresholdAndCompare32sc 
						backends.cuda
						common
						)
target_link_options(mppci_capi_32sc PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_32sc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_32sc PROPERTIES OUTPUT_NAME mppci_capi_32sc)
					

add_library(mppci_capi_32fc SHARED
	mppci_capi_32fc.cpp
	convertScale_32fc.cpp
	copySwapChannelDup_32fc.cpp ) 
target_link_libraries(mppci_capi_32fc PRIVATE project_options project_warnings )
target_link_libraries(mppci_capi_32fc PUBLIC  			
						backends.cuda.image.arithmetic32fc
						backends.cuda.image.dataExchangeAndInit32fc 
						backends.cuda.image.filtering32fc 
						backends.cuda.image.geometryTransforms32fc 
						backends.cuda.image.statistics32fc 
						backends.cuda.image.thresholdAndCompare32fc 
						backends.cuda
						common
						)
target_link_options(mppci_capi_32fc PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>)	
target_include_directories(mppci_capi_32fc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppci_capi_32fc PROPERTIES OUTPUT_NAME mppci_capi_32fc)
					


add_library(mppc_capi_support SHARED mppc_capi_support.cpp ) 

target_link_libraries(mppc_capi_support PRIVATE project_options project_warnings )
target_link_libraries(mppc_capi_support PUBLIC common backends.cuda)
target_include_directories(mppc_capi_support  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src )
set_target_properties(mppc_capi_support PROPERTIES OUTPUT_NAME mppc_capi_support)

