
add_executable(nppCodeGenerator 
				main.cpp
				process8uC1.cpp
				process8uC2.cpp
				process8uC3.cpp
				process8uC4.cpp
				process8sC1.cpp
				process8sC2.cpp
				process8sC3.cpp
				process8sC4.cpp
				process16uC1.cpp
				process16uC2.cpp
				process16uC3.cpp
				process16uC4.cpp
				process16sC1.cpp
				process16sC2.cpp
				process16sC3.cpp
				process16sC4.cpp
				process16scC1.cpp
				process16scC2.cpp
				process16scC3.cpp
				process16scC4.cpp
				process32uC1.cpp
				process32uC2.cpp
				process32uC3.cpp
				process32uC4.cpp
				process32sC1.cpp
				process32sC2.cpp
				process32sC3.cpp
				process32sC4.cpp
				process32scC1.cpp
				process32scC2.cpp
				process32scC3.cpp
				process32scC4.cpp
				process32fC1.cpp
				process32fC2.cpp
				process32fC3.cpp
				process32fC4.cpp
				process64fC1.cpp
				process64fC2.cpp
				process64fC3.cpp
				process64fC4.cpp
				)

target_link_libraries(nppCodeGenerator PRIVATE  
						project_options
						project_warnings)
target_link_libraries(nppCodeGenerator PUBLIC  
						nppParser)
						
target_compile_definitions(nppCodeGenerator PUBLIC -DDEFAULT_OUT_DIR="${PROJECT_SOURCE_DIR}/src/backends/npp")  