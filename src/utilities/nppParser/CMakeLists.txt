
add_executable(nppParser 
				main.cpp
				nppParser.cpp
				)

target_link_libraries(nppParser PRIVATE  
						project_options
						project_warnings)
target_link_libraries(nppParser PUBLIC  
						LIBCLANG::LIBCLANG)