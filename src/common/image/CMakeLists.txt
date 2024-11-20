add_library(common.image 
	border.cpp
	roi.cpp
	size2D.cpp
	sizePitched.cpp
	)

target_link_libraries(common.image PRIVATE  
						project_options
						project_warnings						
						)
						
