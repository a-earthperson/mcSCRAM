set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(MC_SRC
        core/preprocessor.cc
        core/probability_analysis.cc
        core/importance_analysis.cc
)

# Host-only sources: Boost.Math statistics.  Must NOT be listed in
# add_sycl_to_target — AdaptiveCpp CUDA SMCP compiles those TUs as CUDA.
set(MC_HOST_SRC
        scheduler/convergence_policy.cc
        stats/info_gain.cc
)

# Optionally collect public headers for installation
set(MC_HEADERS
        #
        core/direct_eval.h
        #
        event/node.h
        event/sample_shape.h
        #
        kernel/basic_event.h
        kernel/gate.h
        kernel/tally.h
        #
        logger/log_benchmark.h
        logger/log_build.h
        logger/log_compressratio.h
        logger/log_pdag.h
        logger/log_model.h
        logger/log_settings.h
        logger/log_layers.h
        logger/log_tally.h
        logger/log_sample_shaper.h
        logger/csv.h
        #
        prng/philox128.h
        prng/sfc64.h
        prng/state128.h
        prng/wyrand.h
        prng/xorshift128.h
        #
        queue/working_set.h
        queue/layer_manager.h
        queue/layer_manager.tpp
        queue/kernel_builder.h
        queue/sample_shaper.h
        queue/queueable.h
        #
        scheduler/convergence_controller.h
        scheduler/convergence_policy.h
        scheduler/iteration_shape.h
        scheduler/progressbar.h
        #
        stats/ci_utils.h
        stats/diagnostics.h
        stats/info_gain.h
        stats/tally.h
)

# Host-only Boost.Math statistics as a regular (non-SYCL) static library.
# add_sycl_to_target applies the AdaptiveCpp launcher to every source of the
# named target, not only the SOURCES list, so these TUs must live elsewhere.
add_library(mc_host_stats STATIC ${MC_HOST_SRC})
set_target_properties(mc_host_stats PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mc_host_stats
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
)
target_link_libraries(mc_host_stats PUBLIC Boost::math)

# Standalone Catch2 target for the host-only convergence statistic.
# Placed before the Apple early-return so the target is available on all hosts.
# Catch2 is fetched once at the top level (v3.4.0 pin in cmake/FuzzybitsCatch2.cmake).
if(BUILD_TESTING)
  if(NOT TARGET Catch2::Catch2WithMain)
    include("${PROJECT_SOURCE_DIR}/cmake/FuzzybitsCatch2.cmake")
    fuzzybits_ensure_catch2()
  endif()

  add_executable(convergence_stats_tests
    "${PROJECT_SOURCE_DIR}/tests/convergence_stats_tests.cc"
  )
  target_include_directories(convergence_stats_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(convergence_stats_tests PRIVATE
    Catch2::Catch2WithMain
    mc_host_stats
  )
  target_compile_features(convergence_stats_tests PRIVATE cxx_std_23)
  add_test(NAME convergence_stats_tests COMMAND convergence_stats_tests)
  set_property(TEST convergence_stats_tests PROPERTY LABELS convergence unit)
endif()

if(APPLE)
  add_library(mc INTERFACE)
  target_include_directories(mc
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:include/mc>
  )
  target_link_libraries(mc INTERFACE scram AdaptiveCpp::acpp-rt Boost::headers)
  return()
endif()

find_package(_adaptivecpp)
# Add AdaptiveCpp libraries to the global LIBS list
list(APPEND LIBS AdaptiveCpp::acpp-rt)

# Build mc as shared library explicitly (dependencies are static)
add_library(mc SHARED ${MC_SRC} ${MC_HEADERS})

# Mark header files as SYSTEM headers to suppress warnings from third-party code
set_target_properties(mc PROPERTIES PUBLIC_HEADER "${MC_HEADERS}")

# Make sure the compiler can find the mc headers when the library is consumed
target_include_directories(mc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/mc>
)

# Link against the core SCRAM library, system libs, and the AdaptiveCpp SYCL target.
# The imported SYCL::SYCL target is provided by AdaptiveCpp when `find_package(_adaptivecpp)`
# is invoked from the top-level CMakeLists.txt.
# `${LIBS}` comes from the parent scope and contains third-party dependencies.
# Drop Boost.Test: it is a leftover in ${LIBS} from the Boost.Test era and
# would inject a competing `main` into every Catch2 executable that links mc.
set(_MC_LIBS ${LIBS})
list(REMOVE_ITEM _MC_LIBS Boost::unit_test_framework)

target_link_libraries(mc
  PUBLIC
    scram
    ${_MC_LIBS}
    mc_host_stats
)

# Enable AdaptiveCpp device compilation for the library sources
add_sycl_to_target(TARGET mc SOURCES ${MC_SRC})

# Propagate the same debug compile options used by the rest of SCRAM
# so that the new target benefits from the strict checks in Debug builds.

#target_compile_options(mc PRIVATE $<$<CONFIG:DEBUG>:${SCRAM_CXX_FLAGS_DEBUG}>)

# Enforce zero-as-null-pointer warning across all build types
#target_compile_options(mc PRIVATE -Wzero-as-null-pointer-constant)

# Inherit the include directories defined at higher levels (src/ and project root).
# Additional includes specific to this sub-module can be added here if needed.

# Install the library next to the main SCRAM library so that downstream
# consumers can find it in the same prefix. Wheels drop headers and place
# libmc next to the Python extension.
if(SKBUILD)
  fuzzybits_set_wheel_rpath(mc)
  install(
    TARGETS mc
    COMPONENT scram
    LIBRARY DESTINATION fuzzybits
    ARCHIVE DESTINATION fuzzybits
    RUNTIME DESTINATION fuzzybits
  )
else()
  install(
    TARGETS mc
    COMPONENT scram
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mc
  )
endif()