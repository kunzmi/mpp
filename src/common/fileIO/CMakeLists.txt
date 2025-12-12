add_library(common.fileIO SHARED
	file.cpp
	fileIOException.cpp
	fileReader.cpp
	fileWriter.cpp
	lzwCompression.cpp
	pseudoFileReader.cpp
	zlibCompression.cpp

	filetypes/bigTiffImageFileDirectory.cpp
	filetypes/tiffFile.cpp
	filetypes/tiffImageFileDirectory.cpp
	)
	
target_compile_definitions(common.fileIO PUBLIC -DMPP_COMMON_FILEIO)  
	
target_link_libraries(common.fileIO PUBLIC  
						common
						zlib						
						)

target_link_libraries(common.fileIO PRIVATE  
						project_options
						project_warnings						
						)
						
set_target_properties(common.fileIO PROPERTIES OUTPUT_NAME mpp-common-fileIO)