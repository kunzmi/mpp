
add_library(nppParser 
				nppParser.cpp
				)
				
target_link_libraries(nppParser PRIVATE  
						project_options
						project_warnings)
target_link_libraries(nppParser PUBLIC  
						LIBCLANG::LIBCLANG)

list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 NPP_INCL_DIR)
target_compile_definitions(nppParser PRIVATE -DNPP_HEADER_DIR="${NPP_INCL_DIR}")  