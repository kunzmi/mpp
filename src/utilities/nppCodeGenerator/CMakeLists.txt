
add_executable(nppCodeGenerator 
				convertedArgument.cpp
				convertedFunction.cpp
				main.cpp
				)

target_link_libraries(nppCodeGenerator PRIVATE  
						project_options
						project_warnings)
target_link_libraries(nppCodeGenerator PUBLIC  
						nppParser)
						
target_compile_definitions(nppCodeGenerator PUBLIC -DDEFAULT_OUT_DIR="${PROJECT_SOURCE_DIR}/src/backends/npp")  