cmake_minimum_required(VERSION 3.18.4)

# In-source build prevention.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# WIN32: STL Clang-version hatch must apply to *every* nested project(),
# including Boost FetchContent libs/*/CMakeLists.txt. CMAKE_USER_MAKE_RULES_OVERRIDE
# runs at enable_language in each of those; CMAKE_PROJECT_INCLUDE is a second belt.
if(WIN32)
  set(_fb_win_override
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/windows_compiler_override.cmake")
  set(CMAKE_USER_MAKE_RULES_OVERRIDE "${_fb_win_override}" CACHE FILEPATH
      "Windows Clang STL hatch" FORCE)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE_C "${_fb_win_override}" CACHE FILEPATH
      "Windows Clang STL hatch (C)" FORCE)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX "${_fb_win_override}" CACHE FILEPATH
      "Windows Clang STL hatch (C++)" FORCE)
  set(CMAKE_PROJECT_INCLUDE
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/windows_nested_project.cmake")
  unset(_fb_win_override)
endif()

project(SCRAM VERSION 0.18.3 LANGUAGES C CXX)
include(GNUInstallDirs)

# Find and configure LLVM toolchain BEFORE project() call
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(${CMAKE_MODULE_PATH})
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ToolchainPins.cmake")

# CMake 3.27 - Address CMP0144 policy warning
# @link https://cmake.org/cmake/help/latest/policy/CMP0167.html#policy:CMP0144
#
# find_package() uses upper-case <PACKAGENAME>_ROOT variables.
#
# In CMake 3.27 and above the find_package(<PackageName>) command now searches prefixes specified by the upper-case
# <PACKAGENAME>_ROOT CMake variable and the <PACKAGENAME>_ROOT environment variable in addition to the case-preserved
# <PackageName>_ROOT and <PackageName>_ROOT variables used since policy CMP0074. This policy provides compatibility with
# projects that have not been updated to avoid using <PACKAGENAME>_ROOT variables for other purposes.
#
# The OLD behavior for this policy is to ignore <PACKAGENAME>_ROOT variables if the original <PackageName> has
# lower-case characters. The NEW behavior for this policy is to use <PACKAGENAME>_ROOT variables.
#
# This policy was introduced in CMake version 3.27. It may be set by cmake_policy() or cmake_minimum_required(). If it
# is not set, CMake warns, and uses OLD behavior.
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

####################### Begin Options ################### {{{

# Memory allocator. Default is malloc: tcmalloc/jemalloc typically need
# LD_PRELOAD before process start, which cannot work for a pip-installed
# Python module (the interpreter is already running). CLI containers may
# still pass -DMALLOC_TYPE=tcmalloc and set LD_PRELOAD on the entrypoint.
# See docs/build-variants.md.
set(ALLOWED_MALLOC_TYPES "tcmalloc" "jemalloc" "malloc" CACHE STRING "Allowed memory allocator types: tcmalloc, jemalloc, malloc")
set(MALLOC_TYPE "malloc" CACHE STRING "Select the memory allocator type (tcmalloc, jemalloc, malloc)")
option(WITH_COVERAGE "Instrument for coverage analysis" OFF)
option(WITH_PROFILE "Instrument for performance profiling" OFF)
# BUILD_TESTING (from include(CTest) below) is canonical. BUILD_TESTS is the
# legacy option name still passed by Dockerfiles (`-DBUILD_TESTS=OFF`).
option(BUILD_TESTS "Deprecated alias of BUILD_TESTING" ON)
option(OPTIMIZE_FOR_NATIVE "Build with -march=native" ON)
option(BUILD_PYTHON_BINDINGS "Build the native Python extension module" ON)
option(FUZZYBITS_WITH_MC "Build the SYCL Monte Carlo engine (libmc, scram-cli, Python)" ON)

if(SKBUILD)
  # pip/scikit-build-core: skip Catch2 and bake $ORIGIN / @loader_path RPATH
  # into the extension + libscram/libmc so the wheel does not record the
  # build prefix. pcfb + compile + runtime are built into the extension
  # (statically linked); fzb / scram-cli stay out of the wheel (COMPONENT
  # fzb / skipped below). FUZZYBITS_WITH_MC stays ON (need libmc).
  set(BUILD_TESTS OFF CACHE BOOL "Deprecated alias of BUILD_TESTING" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "Build the testing tree." FORCE)
  set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FuzzybitsWheelRpath.cmake")
endif()

if(BUILD_PYTHON_BINDINGS AND MALLOC_TYPE STREQUAL "tcmalloc")
  message(WARNING
    "MALLOC_TYPE=tcmalloc cannot apply to a pip-installed Python module: "
    "tcmalloc must be LD_PRELOAD'd before the interpreter starts. "
    "Use MALLOC_TYPE=malloc (the default) for library/Python builds. "
    "tcmalloc remains valid for scram-cli containers that set LD_PRELOAD.")
endif()

####################### End Options ##################### }}}

####################### Begin compiler configurations ################### {{{

# Default to C++23.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

function(CHECK_COMPILER_VERSION MIN_VERSION)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MIN_VERSION)
    message(FATAL_ERROR "${CMAKE_CXX_COMPILER_ID} Compiler version too old. Required minimum version: ${MIN_VERSION}")
  endif()
endfunction()

add_definitions(-DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")  # Needed to print file paths.
add_definitions(-DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")  # Needed for runtime install path detection.

# Generate version.h from version.h.in
set(SCRAM_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(SCRAM_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(SCRAM_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(SCRAM_VERSION ${PROJECT_VERSION})

# Try to get git revision info
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --always --dirty
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE SCRAM_GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
else()
  set(SCRAM_GIT_REVISION "unknown")
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/src/version.h.in"
  "${PROJECT_BINARY_DIR}/version.h"
  @ONLY
)

# Proactively disable warnings in case Wall/Wextra are enabled outside.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare -Wno-missing-field-initializers -Wno-string-plus-int")

# Strict debug flags for SCRAM targets (opt-in, must subscribe to quality checks explicitly).
# NOTE: This is a list unlike CMAKE_CXX_FLAGS.
#set(SCRAM_CXX_FLAGS_DEBUG -Wall -Wextra -Werror -Wnon-virtual-dtor -Wno-old-style-cast)

if(WIN32)
  # FetchContent Boost + AdaptiveCpp produce command lines that exceed
  # CreateProcess's 32767-char limit unless Ninja uses response files.
  set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _ALLOW_COMPILER_VERSION_MISMATCH)
  # windows-2025 STL (MSVC 14.52+) gates on Clang 22; LLVM 20 is the AdaptiveCpp pin.
  # Directory add_compile_definitions do not reach Boost's nested project();
  # append to CMAKE_<LANG>_FLAGS (keep -DWIN32) and wrap Clang with a launcher.
  if(NOT CMAKE_CXX_FLAGS MATCHES "_ALLOW_COMPILER_VERSION_MISMATCH")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX"
        CACHE STRING "C++ flags" FORCE)
  endif()
  if(NOT CMAKE_C_FLAGS MATCHES "_ALLOW_COMPILER_VERSION_MISMATCH")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX"
        CACHE STRING "C flags" FORCE)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_fb_hatch_py "${PROJECT_SOURCE_DIR}/cmake/windows_clang_hatch_launcher.py")
    find_program(_fb_hatch_python NAMES python3 python python3.12)
    if(_fb_hatch_python AND EXISTS "${_fb_hatch_py}")
      set(CMAKE_C_COMPILER_LAUNCHER "${_fb_hatch_python}" "${_fb_hatch_py}"
          CACHE STRING "Windows Clang STL hatch launcher" FORCE)
      set(CMAKE_CXX_COMPILER_LAUNCHER "${_fb_hatch_python}" "${_fb_hatch_py}"
          CACHE STRING "Windows Clang STL hatch launcher" FORCE)
    endif()
    unset(_fb_hatch_py)
    unset(_fb_hatch_python)
  endif()
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-new-ttp-matching")  # TODO: Boost ICL failure.
  CHECK_COMPILER_VERSION("7.1")

elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  CHECK_COMPILER_VERSION("5.0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-division-by-zero -Wno-nontrivial-memaccess")
  list(APPEND SCRAM_CXX_FLAGS_DEBUG -Wno-missing-braces -Wshadow -Wno-unused-exception-parameter -Wno-sign-compare -Wno-missing-field-initializers -Wno-string-plus-int -Wno-unused-lambda-capture -Wno-old-style-cast -Wno-deprecated-declarations -Wno-unused-parameter)

elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  CHECK_COMPILER_VERSION("9.0")
  list(APPEND SCRAM_CXX_FLAGS_DEBUG -Wno-missing-braces -Wshadow -Wno-unused-exception-parameter -Wno-sign-compare -Wno-missing-field-initializers -Wno-string-plus-int -Wno-unused-lambda-capture -Wno-old-style-cast -Wno-deprecated-declarations -Wno-unused-parameter)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  CHECK_COMPILER_VERSION("18.0.1")
  # TODO: Warning with overload of private override.
  list(APPEND SCRAM_CXX_FLAGS_DEBUG -diag-disable=1125)
endif()

if(WIN32)
  list(APPEND SCRAM_CXX_FLAGS_DEBUG -Wno-error)
  # Shared scram/mc have no explicit dllexport; clang-cl/MSVC needs this to
  # generate import libs. Python wheels use the DLL CRT (/MD), so Boost must
  # not be built against the static CRT.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

if(WITH_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

if(WITH_PROFILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -fno-omit-frame-pointer")
endif()

if(OPTIMIZE_FOR_NATIVE)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
endif()

######################## End compiler configurations #################### }}}

##################### Begin cmake configuration ################### {{{

# Honour -DBUILD_TESTS=OFF from Dockerfiles / legacy callers before CTest
# defines BUILD_TESTING.
if(DEFINED BUILD_TESTS AND NOT BUILD_TESTS)
  set(BUILD_TESTING OFF CACHE BOOL "Build the testing tree." FORCE)
endif()
include(CTest)
if(NOT BUILD_TESTING)
  set(BUILD_TESTS OFF CACHE BOOL "Deprecated alias of BUILD_TESTING" FORCE)
endif()
message(STATUS "BUILD_TESTING=${BUILD_TESTING} (BUILD_TESTS is a deprecated alias)")

if(BUILD_TESTING)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FuzzybitsCatch2.cmake")
  fuzzybits_ensure_catch2()
  include(Catch)
endif()

if(WIN32)
  set(CMAKE_SKIP_RPATH TRUE)
else()
  # Use, i.e. don't skip the full RPATH for the build tree.
  set(CMAKE_SKIP_BUILD_RPATH FALSE)

  # When building,
  # don't use the install RPATH already
  # (but later on when installing).
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

  #set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib/scram")
  #set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib/scram")

  # Add the automatically determined parts of the RPATH,
  # which point to directories outside the build tree
  # to the install RPATH.
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if(SKBUILD)
  set(CMAKE_INSTALL_RPATH "${FUZZYBITS_WHEEL_INSTALL_RPATH}")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
  endif()
endif()

######################## End cmake configuration ################### }}}

## set the memory allocator
# profiling on
if(WITH_PROFILE)
  if(NOT MALLOC_TYPE STREQUAL "malloc")
    message(WARNING "memory allocator: Type ${MALLOC_TYPE} disabled during profiling, using malloc")
  endif()
  message(STATUS "memory allocator: malloc")
  set(MALLOC "System Malloc")
  # profiling off
else()
  # non-system allocator requested
  if(NOT MALLOC_TYPE STREQUAL "malloc")
    # requested tcmalloc
    if(MALLOC_TYPE STREQUAL "tcmalloc")
      find_package(Tcmalloc)
      if(Tcmalloc_FOUND)
        list(APPEND LIBS ${Tcmalloc_LIBRARIES})
        set(MALLOC "TCMalloc")
      else ()
        message(FATAL_ERROR "memory allocator: Requested type tcmalloc not found")
      endif ()
      # requested jemalloc
    elseif (MALLOC_TYPE STREQUAL "jemalloc")
      find_package(JeMalloc)
      if(JEMALLOC_FOUND)
        list(APPEND LIBS ${JEMALLOC_LIBRARIES})
        set(MALLOC "JEMalloc")
      else ()
        message(FATAL_ERROR "memory allocator: Requested type jemalloc not found")
      endif ()
      # requested unsupported allocator
    else()
      message(FATAL_ERROR "memory allocator: Requested type ${MALLOC_TYPE} is invalid")
    endif ()
    # system allocator requested
  else()
    set(MALLOC "System Malloc")
  endif()
  message(STATUS "memory allocator: ${MALLOC_TYPE}")
endif()


# Expose build metadata to C++ via compile definitions
add_compile_definitions(
  SCRAM_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  SCRAM_WITH_COVERAGE=$<BOOL:${WITH_COVERAGE}>
  SCRAM_WITH_PROFILE=$<BOOL:${WITH_PROFILE}>
  SCRAM_OPTIMIZE_NATIVE=$<BOOL:${OPTIMIZE_FOR_NATIVE}>
  SCRAM_MALLOC_TYPE="${MALLOC}"
)

# Find LibXML2 using our custom module
find_package(LibXml2 REQUIRED)
list(APPEND LIBS LibXml2::LibXml2)

list(APPEND LIBS ${CMAKE_DL_LIBS})

# ---------------------------------------------------------------------------
# Boost configuration --------------------------------------------------------
# ---------------------------------------------------------------------------
# Tell the custom FindBoost.cmake which libraries to build/fetch and how.
# The list must be semicolon-separated for CMake variables.
set(BOOST_INCLUDE_LIBRARIES
    "program_options;filesystem;system;random;range;exception;multi_index;accumulators;multiprecision;icl;math;dll;regex;test"
    CACHE STRING "Boost libraries to build via FetchContent" FORCE)

# Enable Boost's official CMake build and request static libs built with PIC.
set(BOOST_ENABLE_CMAKE ON CACHE BOOL "Build Boost using its new CMake build" FORCE)

# Force static libraries for all dependencies to avoid runtime linking issues
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
if(WIN32)
  set(Boost_USE_STATIC_RUNTIME OFF)
else()
  set(Boost_USE_STATIC_RUNTIME ON)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add required Boost libraries
find_package(Boost REQUIRED)

if(WIN32 AND DEFINED boost_SOURCE_DIR)
  # FetchContent Boost is a per-library include tree. Public scram headers
  # include Boost, but scram links Boost PRIVATE, so other TUs do not
  # inherit those -I paths. Glob every libs/*/include (transitive
  # deps like Container are not in BOOST_INCLUDE_LIBRARIES).
  file(GLOB _fb_boost_lib_inc "${boost_SOURCE_DIR}/libs/*/include")
  include_directories(${_fb_boost_lib_inc})
  unset(_fb_boost_lib_inc)
endif()

list(APPEND LIBS
    Boost::program_options
    Boost::filesystem
    Boost::system
    Boost::random
    Boost::range
    Boost::exception
    Boost::multi_index
    Boost::accumulators
    Boost::multiprecision
    Boost::icl
    Boost::math
    Boost::dll
    Boost::regex
    Boost::unit_test_framework
)


########################## End of find libraries ######################## }}}
########################## Begin includes ############################### {{{

# ---- Progress bar (indicators) dependency ----
include(FetchContent)
FetchContent_Declare(
  indicators
  GIT_REPOSITORY https://github.com/p-ranav/indicators.git
  GIT_TAG        v2.3
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(indicators)
include_directories(${indicators_SOURCE_DIR}/include)

if(BUILD_PYTHON_BINDINGS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  # Wheel builds (scikit-build-core) inject the pip-installed pybind11 into
  # CMAKE_PREFIX_PATH. CPython 3.14/3.15 need pybind11 3.x; 3.12 still builds.
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v3.1.0
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()

include_directories("${CMAKE_SOURCE_DIR}")  # Include the core headers via "src".
include_directories("src")  # Include the core headers via "src".
include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR})
include_directories("${PROJECT_BINARY_DIR}/include")

# Generate embedded schemas at configure time
set(CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CURRENT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateEmbeddedSchemas.cmake")

add_subdirectory(src)

# ir::pc + FTP/AIGER IO imported from fuzzybits-reset-2026 (#42).
# Header-only IR and the IO static libs are SYCL-free.
add_subdirectory(src/api)
add_subdirectory(src/ir)
add_subdirectory(src/io)

# pcfb is public; FetchContent uses toolchain.env PCFB_GIT_TAG
# (currently facae79077e49cfdbc3f691c4628999ab16f89b4). Override with
# FUZZYBITS_PCFB_SOURCE_DIR for a local checkout (cmake/FuzzybitsPcfb.cmake).
# Built for both the wheel (SKBUILD) and the in-tree fzb CLI.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FuzzybitsPcfb.cmake")
add_subdirectory(src/compile)
if(FUZZYBITS_WITH_MC)
  add_subdirectory(src/runtime)
endif()
if(NOT SKBUILD)
  add_subdirectory(targets/fzb)
endif()

if(FUZZYBITS_WITH_MC)
  if(NOT SKBUILD)
    add_subdirectory(targets/scram)
  endif()
  if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(targets/python)
  endif()
else()
  message(STATUS "FUZZYBITS_WITH_MC=OFF: skipping scram-cli and Python bindings")
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()