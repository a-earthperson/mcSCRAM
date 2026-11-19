# pcfb is public. Default: FetchContent the pinned v1.1 implementation
# (fuzzybit-org/pcfb@facae790) anonymously. Optional local override:
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
  set(_pcfb_git_tag_default "facae79077e49cfdbc3f691c4628999ab16f89b4")
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
    if(WIN32)
      include("${CMAKE_CURRENT_LIST_DIR}/patches/pcfb_windows_mmap.cmake")
      fuzzybits_pcfb_apply_windows_mmap("${pcfb_SOURCE_DIR}")
    endif()
  endif()

  set(BUILD_TESTING "${_fuzzybits_saved_testing}" CACHE BOOL "Build the testing tree" FORCE)

  if(NOT TARGET pcfb)
    message(FATAL_ERROR "pcfb target was not created")
  endif()
  if(NOT TARGET pcfb::pcfb)
    add_library(pcfb::pcfb ALIAS pcfb)
  endif()
endfunction()

fuzzybits_add_pcfb()
