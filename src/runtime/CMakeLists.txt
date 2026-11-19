# pcfb → PDAG reconstruction (host) + Monte Carlo run (SYCL).
#
# Reconstruction is a regular TU: AdaptiveCpp CUDA SMCP compiles every
# source of an add_sycl_to_target'd library as CUDA (#60).
#
# The SYCL entrypoints (run.cc) are compiled into libmc so they inherit
# AdaptiveCpp::acpp-rt from src/mc. Calling add_sycl_to_target from this
# directory fails: those imported targets are directory-scoped to src/mc,
# and promoting them IMPORTED_GLOBAL breaks targets/python's second
# find_package(AdaptiveCpp).

add_library(fuzzybits_runtime_host STATIC
  pdag_from_pcfb.cc
)
set_target_properties(fuzzybits_runtime_host PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME fuzzybits_runtime_host
)
target_include_directories(fuzzybits_runtime_host
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_link_libraries(fuzzybits_runtime_host
  PUBLIC
    scram
    pcfb
)

# Device-side Monte Carlo.
# RULE_LAUNCH_COMPILE / CUDA SMCP apply to every source of mc. PRIVATE link
# of the host archive so its objects land in libmc.so once (fzb / tests
# must not also pull the .a, or pdag_from_circuit is defined twice).
#
# Darwin has no separate libmc.dylib (INTERFACE mc → libscram). Linking
# fuzzybits_runtime_host into scram would cycle (the .a PUBLIC-links scram),
# so compile the TUs into libscram instead.
if(APPLE)
  # run.cc / pdag_from_pcfb.cc are compiled into libscram in src/CMakeLists.txt
  # (add_sycl_to_target must run in the directory that found AdaptiveCpp).
  target_link_libraries(scram PRIVATE pcfb)
  target_link_libraries(mc INTERFACE pcfb)
else()
  target_sources(mc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/run.cc")
  target_link_libraries(mc PRIVATE fuzzybits_runtime_host)
  target_link_libraries(mc PUBLIC pcfb)
endif()

add_library(fuzzybits_runtime INTERFACE)
target_include_directories(fuzzybits_runtime INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_link_libraries(fuzzybits_runtime INTERFACE mc)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
