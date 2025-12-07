add_library(common.image OBJECT
	affineTransformation.cpp
	border.cpp
	matrix.cpp
	pixelTypes.cpp
	roi.cpp
	roiException.cpp
	size2D.cpp
	sizePitched.cpp
	solve.cpp
	)

target_compile_definitions(common.image PUBLIC -DMPP_COMMON)  
	
#target_link_libraries(common.image PUBLIC  
#						common						
#						)

target_link_libraries(common.image PRIVATE  
						project_options
						project_warnings						
						)
						
