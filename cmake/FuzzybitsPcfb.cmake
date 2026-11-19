# pcfb is public. Default: FetchContent the pinned v1.1 implementation
# (fuzzybit-org/pcfb@6b1af8bb) anonymously. Optional local override:
#   -DFUZZYBITS_PCFB_SOURCE_DIR=/path/to/pcfb
# (Dockerfile.fzb still accepts --build-context pcfb=… for offline builds.)
#
# Pin: toolchain.env PCFB_GIT_TAG → FUZZYBITS_PIN_PCFB_GIT_TAG (#59). The
# CACHE default below matches that pin so a configure without ToolchainPins
# still FetchContents the same revision.

set(FUZZYBITS_PCFB_SOURCE_DIR "" CACHE PATH
    "Path to a local pcfb checkout (skips FetchContent)")
if(DEFINED FUZZYBITS_PIN_PCFB_GIT_TAG AND NOT FUZZYBITS_PIN_PCFB_GIT_TAG STREQUAL "")
  set(_pcfb_git_tag_default "${FUZZYBITS_PIN_PCFB_GIT_TAG}")
else()
  set(_pcfb_git_tag_default "6b1af8bb4bb9af3d217a33317c006f0ff9516cf1")
endif()
set(FUZZYBITS_PCFB_GIT_TAG "${_pcfb_git_tag_default}" CACHE STRING
    "Pinned pcfb git revision (v1.1)")
unset(_pcfb_git_tag_default)

if(DEFINED FUZZYBITS_PIN_PCFB_GIT_REPO AND NOT FUZZYBITS_PIN_PCFB_GIT_REPO STREQUAL "")
  set(_pcfb_git_repo_default "${FUZZYBITS_PIN_PCFB_GIT_REPO}")
else()
  set(_pcfb_git_repo_default "https://github.com/fuzzybit-org/pcfb.git")
endif()
set(FUZZYBITS_PCFB_GIT_REPO "${_pcfb_git_repo_default}" CACHE STRING
    "pcfb git remote (ignored when FUZZYBITS_PCFB_SOURCE_DIR is set)")
unset(_pcfb_git_repo_default)

function(fuzzybits_add_pcfb)
  if(TARGET pcfb)
    return()
  endif()

  # pcfb's CMakeLists enables its own Catch2 suite on BUILD_TESTING. Keep that
  # off while we import it — fuzzybits tests live under src/compile/tests.
  # include(CTest) inside pcfb would otherwise reset the cache option to ON.
  set(_fuzzybits_saved_testing "${BUILD_TESTING}")
  set(BUILD_TESTING OFF CACHE BOOL "Build the testing tree" FORCE)
  set(PCFB_BUILD_TESTS OFF CACHE BOOL "Build pcfb tests" FORCE)

  if(FUZZYBITS_PCFB_SOURCE_DIR)
    if(NOT EXISTS "${FUZZYBITS_PCFB_SOURCE_DIR}/CMakeLists.txt")
      message(FATAL_ERROR
        "FUZZYBITS_PCFB_SOURCE_DIR=${FUZZYBITS_PCFB_SOURCE_DIR} does not contain CMakeLists.txt")
    endif()
    message(STATUS "pcfb: add_subdirectory(${FUZZYBITS_PCFB_SOURCE_DIR}) pin ${FUZZYBITS_PCFB_GIT_TAG}")
    add_subdirectory("${FUZZYBITS_PCFB_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/pcfb-build"
                     EXCLUDE_FROM_ALL)
  else()
    message(STATUS "pcfb: FetchContent ${FUZZYBITS_PCFB_GIT_REPO} @ ${FUZZYBITS_PCFB_GIT_TAG}")
    include(FetchContent)
    FetchContent_Declare(pcfb
      GIT_REPOSITORY ${FUZZYBITS_PCFB_GIT_REPO}
      GIT_TAG        ${FUZZYBITS_PCFB_GIT_TAG}
    )
    FetchContent_MakeAvailable(pcfb)
  endif()

  set(BUILD_TESTING "${_fuzzybits_saved_testing}" CACHE BOOL "Build the testing tree" FORCE)

  if(NOT TARGET pcfb)
    message(FATAL_ERROR "pcfb target was not created")
  endif()
  if(NOT TARGET pcfb::pcfb)
    add_library(pcfb::pcfb ALIAS pcfb)
  endif()

  # pcfb cmake/common-flags.cmake does add_compile_options(-march=native).
  # That bakes the *builder* CPU into the static lib (GHA: BMI2 `bzhi`;
  # xerxes: Ivy Bridge, so a xerxes-side objdump is a false pass). Replace
  # the directory COMPILE_OPTIONS snapshot and pin -mno-bmi2 after any
  # leftover -march=native so the last ABI-affecting flags are portable.
  if(NOT OPTIMIZE_FOR_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(NOT DEFINED FUZZYBITS_X86_ARCH OR FUZZYBITS_X86_ARCH STREQUAL "")
      set(FUZZYBITS_X86_ARCH "x86-64-v2")
    endif()
    set(_fb_portable_march
      "-march=${FUZZYBITS_X86_ARCH}"
      "-mno-avx2" "-mno-fma" "-mno-bmi" "-mno-bmi2")
    if(FUZZYBITS_PCFB_SOURCE_DIR)
      set(_pcfb_dir "${FUZZYBITS_PCFB_SOURCE_DIR}")
    else()
      FetchContent_GetProperties(pcfb SOURCE_DIR _pcfb_dir)
    endif()
    foreach(_dir "${_pcfb_dir}" "${_pcfb_dir}/src")
      if(IS_DIRECTORY "${_dir}")
        set_property(DIRECTORY "${_dir}" PROPERTY COMPILE_OPTIONS "${_fb_portable_march}")
      endif()
    endforeach()
    FetchContent_GetProperties(flatbuffers SOURCE_DIR _flatbuffers_dir)
    if(IS_DIRECTORY "${_flatbuffers_dir}")
      set_property(DIRECTORY "${_flatbuffers_dir}" PROPERTY COMPILE_OPTIONS "${_fb_portable_march}")
    endif()
    foreach(_t pcfb flatbuffers flatc)
      if(TARGET ${_t})
        target_compile_options(${_t} PRIVATE ${_fb_portable_march})
      endif()
    endforeach()
    message(STATUS "pcfb portable x86: ${_fb_portable_march}")
    unset(_fb_portable_march)
    unset(_pcfb_dir)
    unset(_flatbuffers_dir)
    unset(_dir)
    unset(_t)
  endif()
endfunction()

fuzzybits_add_pcfb()
