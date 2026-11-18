fuzzybits_ensure_catch2()
fuzzybits_add_imported_test(
  TARGET fzb_runtime_tests
  SOURCES runtime_tests.cc
  LINK_LIBRARIES fuzzybits_runtime fuzzybits_compile
)
set_tests_properties(fzb_runtime_tests PROPERTIES
  LABELS "mc"
  TIMEOUT 120
)
