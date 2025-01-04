option(ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" ON)
option(ENABLE_INCLUDE_WHAT_YOU_USE "Enable static analysis with include-what-you-use" OFF)

if(ENABLE_CPPCHECK)
  find_program(CPPCHECK cppcheck HINT ${CPPCHECK_HINT_DIR})
  if(CPPCHECK)
    set(CMAKE_CXX_CPPCHECK
        ${CPPCHECK}
        --suppress=missingInclude
        --enable=all
        --inline-suppr
        --inconclusive
        -i
        ${CMAKE_SOURCE_DIR}/imgui/lib)
  else()
    message(SEND_ERROR "cppcheck requested but executable not found")
  endif()
endif()

if(ENABLE_CLANG_TIDY)
  find_program(CLANGTIDY NAMES clang-tidy-18 clang-tidy HINTS ${CLANGTIDY_HINT_DIR})
  if(CLANGTIDY)
    if(MSVC)
	  #MSVC needs extra exception parameter EHsc
	  set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY};-header-filter=.*src[\\|\/][libraries|applications].*;--extra-arg=/EHsc;)	
	else()
      set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY};-header-filter=.*src[\\|\/][libraries|applications].*;--extra-arg=-Wno-unknown-warning-option;)
    endif()
  else()
    message(SEND_ERROR "clang-tidy requested but executable not found")
  endif()
endif()

if(ENABLE_INCLUDE_WHAT_YOU_USE)
  find_program(INCLUDE_WHAT_YOU_USE include-what-you-use HINT ${INCLUDE_WHAT_YOU_USE_HINT_DIR})
  if(INCLUDE_WHAT_YOU_USE)
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${INCLUDE_WHAT_YOU_USE})
  else()
    message(SEND_ERROR "include-what-you-use requested but executable not found")
  endif()
endif()
