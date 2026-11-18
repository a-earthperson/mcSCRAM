fuzzybits_ensure_catch2()

fuzzybits_add_imported_test(
  TARGET fuzzybits_ir_pc_core_tests
  SOURCES pc_core_tests.cc
  LINK_LIBRARIES fuzzybits_ir_pc
)

fuzzybits_add_imported_test(
  TARGET fuzzybits_ir_pc_usage_smoke
  SOURCES usage_smoke.cc
  LINK_LIBRARIES fuzzybits_ir_pc
)
