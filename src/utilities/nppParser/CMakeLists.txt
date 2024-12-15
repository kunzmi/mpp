
add_library(nppParser 
				nppParser.cpp
				)
				
target_link_libraries(nppParser PRIVATE  
						project_options
						project_warnings)
target_link_libraries(nppParser PUBLIC  
						LIBCLANG::LIBCLANG)

target_compile_definitions(nppParser PUBLIC -DNPP_HEADER_DIR="${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")  