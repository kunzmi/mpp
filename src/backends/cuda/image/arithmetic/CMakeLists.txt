add_library(backends.cuda.image.arithmetic 
	abs.cu

	absDiff.cu

	add.cu
	addMasked.cu
	
	addProduct.cu
	addProductMasked.cu
	
	addSquare.cu
	addSquareMasked.cu
	
	addWeighted.cu
	addWeightedMasked.cu
	
	alphaComp.cu
	alphaCompC.cu
	alphaPremul.cu

	and.cu
	
	angle.cu
	
	conj.cu

	conjMul.cu

	div.cu
	divMasked.cu
	
	exp.cu
	
	imag.cu

	ln.cu
	
	lshift.cu
	
	magnitude.cu
	
	magnitudeSqr.cu
	
	makeComplex.cu

	mul.cu
	mulMasked.cu
	
	not.cu

	or.cu
	
	real.cu
	
	rshift.cu

	sqr.cu

	sqrt.cu

	sub.cu
	subMasked.cu

	xor.cu
	)
	
target_include_directories(backends.cuda.image.arithmetic  PUBLIC 
						   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
						   ${PROJECT_SOURCE_DIR}/src
						  )
target_link_libraries(backends.cuda.image.arithmetic PRIVATE  
						project_options					
						)
target_link_libraries(backends.cuda.image.arithmetic PUBLIC  
						backends.cuda
						common
						common.image
						)