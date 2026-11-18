# Host-only OpenPSA → pcfb lowering. No SYCL.
add_library(fuzzybits_compile STATIC
  pcfb_writer.cc
  exact_eval.cc
  sha256.cc
)
set_target_properties(fuzzybits_compile PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME fuzzybits_compile
)
target_include_directories(fuzzybits_compile
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_link_libraries(fuzzybits_compile
  PUBLIC
    scram
    pcfb
  PRIVATE
    ${LIBS}
)

# CustomPreprocessor<DirectEval>::Run lives in src/mc/core/preprocessor.cc.
# On Apple it is already compiled into libscram; on Linux it lives in libmc
# (SYCL). Compile it here so `fzb` does not need AdaptiveCpp.
if(NOT APPLE)
  target_sources(fuzzybits_compile PRIVATE "${PROJECT_SOURCE_DIR}/src/mc/core/preprocessor.cc")
  target_include_directories(fuzzybits_compile PRIVATE "${PROJECT_SOURCE_DIR}/src/mc")
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
