
add_executable(nppFunctionLister 
				main.cpp
				)

target_link_libraries(nppFunctionLister PRIVATE  
						project_options
						project_warnings)
target_link_libraries(nppFunctionLister PUBLIC  
						nppParser)
						
target_compile_definitions(nppFunctionLister PUBLIC -DDEFAULT_OUT_DIR="${CMAKE_CURRENT_BINARY_DIR}")  