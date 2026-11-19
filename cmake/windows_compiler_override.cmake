# CMAKE_USER_MAKE_RULES_OVERRIDE. Included while enabling C/CXX in *every*
# project(), including FetchContent Boost (libs/container/project()) and the
# LLVM-in-tree AdaptiveCpp configure. Windows + Clang only (clang / clang-cl).
# Must define _ALLOW_COMPILER_VERSION_MISMATCH before yvals_core.h (STL1000
# on MSVC 14.52+ vs LLVM 20).
if(WIN32)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR "${CMAKE_C_COMPILER}" MATCHES "[Cc]lang")
    if(NOT CMAKE_C_FLAGS MATCHES "_ALLOW_COMPILER_VERSION_MISMATCH")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX"
          CACHE STRING "C flags" FORCE)
    endif()
    string(APPEND CMAKE_C_FLAGS_INIT " -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR "${CMAKE_CXX_COMPILER}" MATCHES "[Cc]lang")
    if(NOT CMAKE_CXX_FLAGS MATCHES "_ALLOW_COMPILER_VERSION_MISMATCH")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX"
          CACHE STRING "C++ flags" FORCE)
    endif()
    string(APPEND CMAKE_CXX_FLAGS_INIT " -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX")
  endif()
endif()
