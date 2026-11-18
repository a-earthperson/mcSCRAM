fuzzybits_ensure_catch2()
fuzzybits_add_imported_test(
  TARGET fzb_compile_tests
  SOURCES compile_tests.cc
  LINK_LIBRARIES fuzzybits_compile
)
if(NOT APPLE)
  target_sources(fzb_compile_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/compile/direct_eval_host_stubs.cc")
  target_include_directories(fzb_compile_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/mc")
endif()
# Aralia models are large enough that exhaustive eval of chinese (~25 events)
# needs more than Catch's default 5s.
set_tests_properties(fzb_compile_tests PROPERTIES TIMEOUT 120)
