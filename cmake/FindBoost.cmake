# FindBoost.cmake
# Custom finder that retrieves Boost via FetchContent when the standard
# imported targets are not already available.  It is intentionally minimal –
# for projects that just need Boost’s CMake targets rather than the classic
# FindBoost variables.

# ---------------------------------------------------------------------------
# Fast-exit if Boost targets are already present (e.g. another call fetched it
# earlier in this configure run, or the user supplies their own installation
# via a toolchain file).
# ---------------------------------------------------------------------------
if(TARGET Boost::headers)
    set(Boost_FOUND TRUE)
    return()
endif()

# Prefer a CMake-package Boost (the #33 toolchain image installs 1.89.0 into
# /usr/local) over FetchContent. CONFIG + EXACT pin so a newer system Boost
# (Homebrew 1.90, etc.) is not silently picked up; mismatch falls through to
# FetchContent of FUZZYBITS_PIN_BOOST_VERSION. CONFIG mode does not re-enter
# this module. BoostConfig only creates Boost::<lib> imported targets when
# COMPONENTS are requested; a bare find_package(Boost CONFIG) yields
# Boost::headers.
if(NOT TARGET Boost::program_options)
    if(DEFINED FUZZYBITS_PIN_BOOST_VERSION AND FUZZYBITS_PIN_BOOST_VERSION)
        set(_fuzzybits_boost_ver "${FUZZYBITS_PIN_BOOST_VERSION}")
    else()
        set(_fuzzybits_boost_ver "1.89.0")
    endif()
    set(_fuzzybits_boost_comps ${BOOST_INCLUDE_LIBRARIES})
    if(_fuzzybits_boost_comps)
        list(TRANSFORM _fuzzybits_boost_comps REPLACE "^test$" "unit_test_framework")
        # Boost.DLL is header-only and the 1.89 CMake tarball does not install a
        # boost_dll-config package; do not require it as a CONFIG component.
        list(REMOVE_ITEM _fuzzybits_boost_comps dll)
        find_package(Boost ${_fuzzybits_boost_ver} EXACT CONFIG COMPONENTS ${_fuzzybits_boost_comps} QUIET)
    else()
        find_package(Boost ${_fuzzybits_boost_ver} EXACT CONFIG QUIET)
    endif()
    unset(_fuzzybits_boost_ver)
    unset(_fuzzybits_boost_comps)
endif()
if(TARGET Boost::headers AND TARGET Boost::program_options)
    set(Boost_FOUND TRUE)
    get_target_property(_boost_include Boost::headers INTERFACE_INCLUDE_DIRECTORIES)
    set(Boost_INCLUDE_DIRS ${_boost_include})
    unset(Boost_LIBRARIES)
    if(BOOST_INCLUDE_LIBRARIES)
        foreach(_lib IN LISTS BOOST_INCLUDE_LIBRARIES)
            string(STRIP "${_lib}" _lib)
            if(_lib STREQUAL "test")
                set(_lib unit_test_framework)
            endif()
            if(TARGET Boost::${_lib})
                list(APPEND Boost_LIBRARIES Boost::${_lib})
            endif()
        endforeach()
    endif()
    # Boost.DLL is header-only; Boost 1.89's CMake package does not export
    # Boost::dll. Synthesize it so target_link_libraries(Boost::dll) works.
    if(NOT TARGET Boost::dll AND TARGET Boost::headers)
        add_library(Boost::dll INTERFACE IMPORTED)
        if(TARGET Boost::filesystem)
            target_link_libraries(Boost::dll INTERFACE Boost::headers Boost::filesystem)
        else()
            target_link_libraries(Boost::dll INTERFACE Boost::headers)
        endif()
    endif()
    return()
endif()

# ---------------------------------------------------------------------------
# Fetch Boost
# ---------------------------------------------------------------------------
include(FetchContent)

# Silence nested warnings triggered by calls to
# find_package_handle_standard_args() inside Boost’s sub-projects.
set(_scram_prev_fphsa_name_mismatched ${FPHSA_NAME_MISMATCHED})
set(FPHSA_NAME_MISMATCHED TRUE)

# Save current developer-warning setting so that we can restore it later –
# Boost’s build system emits a handful of dev-level warnings that we don’t
# want to propagate to users of this project.
set(_scram_prev_warn_dev ${CMAKE_WARN_DEV})
set(CMAKE_WARN_DEV FALSE)

# Allow callers to specify which Boost libraries to build by defining the
# BOOST_INCLUDE_LIBRARIES cache variable *before* the outer find_package()
# call.  We do not alter it here – we merely ensure it is cached so that the
# value survives through the FetchContent configure step.
if(NOT DEFINED BOOST_INCLUDE_LIBRARIES)
    set(BOOST_INCLUDE_LIBRARIES "" CACHE STRING "Boost libraries to build via FetchContent")
endif()

# Boost's CMake build is enabled by the BOOST_ENABLE_CMAKE option.
set(BOOST_ENABLE_CMAKE ON CACHE BOOL "Build Boost using its new CMake build" FORCE)

# Force static libraries if BUILD_SHARED_LIBS is OFF
if(NOT BUILD_SHARED_LIBS)
    set(Boost_USE_STATIC_LIBS ON)
    # Additional Boost-specific options to ensure static builds.
    # Windows Python extensions link the DLL CRT (/MD); Boost's "static"
    # runtime is /MT and would mix CRTs.
    if(NOT WIN32)
        set(BOOST_RUNTIME_LINK "static" CACHE STRING "Boost runtime linking")
    endif()
endif()

# Position-independent code is generally required when building shared
# libraries that depend on Boost.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Force static libraries for Boost build when BUILD_SHARED_LIBS is OFF
if(NOT BUILD_SHARED_LIBS)
    # Save current BUILD_SHARED_LIBS state
    set(_boost_build_shared_libs_backup ${BUILD_SHARED_LIBS})
    # Force Boost to build static libraries
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force Boost static libraries" FORCE)
    # Additional Boost-specific settings for static builds
    set(Boost_USE_STATIC_LIBS ON CACHE BOOL "Use static Boost libraries" FORCE)
    if(WIN32)
        set(Boost_USE_STATIC_RUNTIME OFF CACHE BOOL "Use static runtime" FORCE)
    else()
        set(Boost_USE_STATIC_RUNTIME ON CACHE BOOL "Use static runtime" FORCE)
    endif()
endif()

# Boost 1.87/1.88 incorrectly enable GPU/SYCL code paths in Boost.Math, which
# breaks AdaptiveCpp CUDA SMCP (clang -x cuda):
#   error: __host__ function 'iround' cannot overload __host__ __device__ function
# AdaptiveCpp docs: use Boost < 1.87 or >= 1.89. 1.89 also ships the CMake tarball.
FetchContent_Declare(
    boost
    URL      https://github.com/boostorg/boost/releases/download/boost-1.89.0/boost-1.89.0-cmake.tar.gz
    URL_HASH SHA256=954a01219bf818c7fb850fa610c2c8c71a4fa28fa32a1900056bcb6ff58cf908
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(boost)

# Boost.Math must not be included from translation units passed to
# add_sycl_to_target (AdaptiveCpp CUDA SMCP sets __CUDACC__ and Boost 1.87+
# then enables broken GPU overloads of iround). Host-only statistics live in
# src/mc/scheduler/convergence_policy.cc and src/mc/stats/info_gain.cc.

# Restore BUILD_SHARED_LIBS if it was changed
if(DEFINED _boost_build_shared_libs_backup)
    set(BUILD_SHARED_LIBS ${_boost_build_shared_libs_backup} CACHE BOOL "Build shared libraries" FORCE)
    unset(_boost_build_shared_libs_backup)
endif()

# Restore developer-warning and FPHSA settings.
set(CMAKE_WARN_DEV ${_scram_prev_warn_dev})
if(DEFINED _scram_prev_fphsa_name_mismatched)
    set(FPHSA_NAME_MISMATCHED ${_scram_prev_fphsa_name_mismatched})
else()
    unset(FPHSA_NAME_MISMATCHED)
endif()

# ---------------------------------------------------------------------------
# Verify that Boost targets now exist and expose the conventional variables
# expected by downstream code.
# ---------------------------------------------------------------------------
if(TARGET Boost::headers)
    set(Boost_FOUND TRUE)
    # Expose helper variables for projects that still rely on them.
    get_target_property(_boost_include Boost::headers INTERFACE_INCLUDE_DIRECTORIES)
    set(Boost_INCLUDE_DIRS ${_boost_include})
    # Collect all requested component libraries (if any were listed).
    unset(Boost_LIBRARIES)
    if(BOOST_INCLUDE_LIBRARIES)
        foreach(_lib IN LISTS BOOST_INCLUDE_LIBRARIES)
            string(STRIP "${_lib}" _lib)
            # Boost target naming convention is Boost::<component>
            if(TARGET Boost::${_lib})
                list(APPEND Boost_LIBRARIES Boost::${_lib})
            endif()
        endforeach()
    endif()
else()
    set(Boost_FOUND FALSE)
    message(FATAL_ERROR "Boost targets were not created by FetchContent – check FetchContent logs for details.")
endif() 