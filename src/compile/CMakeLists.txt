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
# On Apple it is already compiled into libscram. On Linux it lives in libmc
# (SYCL). Compile it here only for the host-only path — otherwise both
# fuzzybits_compile and libmc define Run (different flags → ODR).
if(NOT APPLE AND NOT FUZZYBITS_WITH_MC)
  target_sources(fuzzybits_compile PRIVATE "${PROJECT_SOURCE_DIR}/src/mc/core/preprocessor.cc")
  target_include_directories(fuzzybits_compile PRIVATE "${PROJECT_SOURCE_DIR}/src/mc")
endif()

if(FUZZYBITS_WITH_MC)
  target_link_libraries(fuzzybits_compile PUBLIC mc)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
