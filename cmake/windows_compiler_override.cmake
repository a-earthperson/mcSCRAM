# CMAKE_USER_MAKE_RULES_OVERRIDE. Included while enabling C/CXX in *every*
# project(), including FetchContent Boost and the LLVM-in-tree AdaptiveCpp
# configure. Windows + Clang only (clang / clang-cl).
#
# Append to FLAGS_INIT only — never CACHE FORCE CMAKE_<LANG>_FLAGS. An explicit
# CMAKE_CXX_FLAGS cache value replaces CMake's Windows INIT (which provides
# -DWIN32); AdaptiveCpp's omp_queue.cpp is gated on WIN32, not _WIN32.
if(WIN32)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR "${CMAKE_C_COMPILER}" MATCHES "[Cc]lang")
    if(NOT CMAKE_C_FLAGS_INIT MATCHES "_ALLOW_COMPILER_VERSION_MISMATCH")
      string(APPEND CMAKE_C_FLAGS_INIT " -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX")
    endif()
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR "${CMAKE_CXX_COMPILER}" MATCHES "[Cc]lang")
    if(NOT CMAKE_CXX_FLAGS_INIT MATCHES "_ALLOW_COMPILER_VERSION_MISMATCH")
      string(APPEND CMAKE_CXX_FLAGS_INIT " -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX")
    endif()
  endif()
endif()
