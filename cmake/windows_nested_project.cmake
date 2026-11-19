# Included from every project() on WIN32 via CMAKE_PROJECT_INCLUDE.
# Directory compile definitions do not cross Boost's per-lib project()
# calls; CACHE FORCE CMAKE_<LANG>_FLAGS so later TUs still see the hatch.
# Clang only — MSVC cl.exe does not need this (and AdaptiveCpp cannot use cl).
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "Clang"
    OR "${CMAKE_CXX_COMPILER}" MATCHES "[Cc]lang")
  add_compile_definitions(_ALLOW_COMPILER_VERSION_MISMATCH NOMINMAX)
  if(NOT CMAKE_CXX_FLAGS MATCHES "_ALLOW_COMPILER_VERSION_MISMATCH")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX"
        CACHE STRING "C++ flags" FORCE)
  endif()
  if(NOT CMAKE_C_FLAGS MATCHES "_ALLOW_COMPILER_VERSION_MISMATCH")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX"
        CACHE STRING "C flags" FORCE)
  endif()
endif()
