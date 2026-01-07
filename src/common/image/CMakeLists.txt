add_library(common.image OBJECT
	affineTransformation.cpp
	border.cpp
	matrix.cpp
	matrix3x4.cpp
	matrix4x4.cpp
	pixelTypes.cpp
	roi.cpp
	roiException.cpp
	size2D.cpp
	sizePitched.cpp
	solve.cpp
	)

target_compile_definitions(common.image PRIVATE -DMPP_COMMON)  
	
#target_link_libraries(common.image PUBLIC  
#						common						
#						)

target_link_libraries(common.image PRIVATE  
						project_options
						project_warnings						
						)
						
