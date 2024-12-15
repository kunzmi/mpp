add_library(common.fileIO 
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
	
target_link_libraries(common.fileIO PUBLIC  
						common.image
						zlib						
						)

target_link_libraries(common.fileIO PRIVATE  
						project_options
						project_warnings						
						)
						