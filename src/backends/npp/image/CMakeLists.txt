add_library(backends.npp.image OBJECT
	image8s.cpp
	image8u.cpp
	image16s.cpp
	image16sc.cpp
	image16u.cpp
	image32s.cpp
	image32sc.cpp
	image32u.cpp
	image16f.cpp
	image32f.cpp
	image32fc.cpp
	image64f.cpp
	image8sC1View.cpp
	image8sC2View.cpp
	image8sC3View.cpp
	image8sC4View.cpp
	image8uC1View.cpp
	image8uC2View.cpp
	image8uC3View.cpp
	image8uC4View.cpp
	image16sC1View.cpp
	image16sC2View.cpp
	image16sC3View.cpp
	image16sC4View.cpp
	image16scC1View.cpp
	image16scC2View.cpp
	image16scC3View.cpp
	image16scC4View.cpp
	image16uC1View.cpp
	image16uC2View.cpp
	image16uC3View.cpp
	image16uC4View.cpp
	image16fC1View.cpp
	image16fC2View.cpp
	image16fC3View.cpp
	image16fC4View.cpp
	image32fC1View.cpp
	image32fC2View.cpp
	image32fC3View.cpp
	image32fC4View.cpp
	image32fcC1View.cpp
	image32fcC2View.cpp
	image32fcC3View.cpp
	image32fcC4View.cpp
	image32sC1View.cpp
	image32sC2View.cpp
	image32sC3View.cpp
	image32sC4View.cpp
	image32scC1View.cpp
	image32scC2View.cpp
	image32scC3View.cpp
	image32scC4View.cpp
	image32uC1View.cpp
	image32uC2View.cpp
	image32uC3View.cpp
	image32uC4View.cpp
	image64fC1View.cpp
	image64fC2View.cpp
	image64fC3View.cpp
	image64fC4View.cpp
	)

target_compile_definitions(backends.npp.image PRIVATE -DMPP_NPP)  
	
target_include_directories(backends.npp.image PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.npp.image PRIVATE  
						project_options
						project_warnings						
						)
target_link_libraries(backends.npp.image PUBLIC 
						)
