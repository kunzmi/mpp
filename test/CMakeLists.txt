
#add_library(catch_main STATIC main.cpp)
#target_link_libraries(catch_main PUBLIC catch2)
#target_link_libraries(catch_main PRIVATE project_options)

set(DEFAULT_CUDA_DEVICE_ID 0)

function(addCatchTest sourceFiles libNames testName)
	add_executable(test_${testName} ${sourceFiles})
	target_link_libraries(test_${testName} PRIVATE project_warnings project_options Catch2::Catch2WithMain ${libNames})
	target_compile_definitions(test_${testName} PUBLIC -DTEST_DATA_DIR="${PROJECT_SOURCE_DIR}/test/testData/")  
	target_compile_definitions(test_${testName} PUBLIC -DDEFAULT_CUDA_DEVICE_ID=${DEFAULT_CUDA_DEVICE_ID})  
	target_include_directories(test_${testName} PUBLIC ${PROJECT_SOURCE_DIR}/test/catch2/src)

	# automatically discover tests that are defined in catch based test files you can modify the unittests. Set TEST_PREFIX
	# to whatever you want, or use different for different binaries
	catch_discover_tests(
	  test_${testName}
	  TEST_PREFIX
	  "${testName}."
	  REPORTER
	  xml
	  OUTPUT_DIR
	  .
	  OUTPUT_PREFIX
	  "${testName}."
	  OUTPUT_SUFFIX
	  .xml)
endfunction()

add_subdirectory(tests)