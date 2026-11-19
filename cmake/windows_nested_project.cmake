# Included from every project() on WIN32 via CMAKE_PROJECT_INCLUDE.
# FetchContent (Boost, libxml2, pcfb) calls project() and drops directory
# compile definitions; CMAKE_CXX_FLAGS from the superbuild is not always
# inherited. -D must be on the command line before yvals_core.h (STL1000).
add_compile_definitions(_ALLOW_COMPILER_VERSION_MISMATCH NOMINMAX)
if(NOT CMAKE_CXX_FLAGS MATCHES "_ALLOW_COMPILER_VERSION_MISMATCH")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX")
endif()
if(NOT CMAKE_C_FLAGS MATCHES "_ALLOW_COMPILER_VERSION_MISMATCH")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX")
endif()
