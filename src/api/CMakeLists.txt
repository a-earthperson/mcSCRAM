# Leaf API types used by ir::pc and FTP/AIGER IO. This is not the
# role/provider/registry/pipeline framework from the May-2026 reset.

add_library(fuzzybits_api INTERFACE)
target_include_directories(fuzzybits_api INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

include("${PROJECT_SOURCE_DIR}/cmake/FuzzybitsCatch2.cmake")

function(fuzzybits_add_imported_test)
  fuzzybits_ensure_catch2()
  set(one_value_args TARGET)
  set(multi_value_args SOURCES LINK_LIBRARIES LABELS)
  cmake_parse_arguments(PARSE_ARGV 0 FB_TEST "" "${one_value_args}" "${multi_value_args}")

  add_executable(${FB_TEST_TARGET} ${FB_TEST_SOURCES})
  target_link_libraries(${FB_TEST_TARGET} PRIVATE ${FB_TEST_LINK_LIBRARIES} Catch2::Catch2WithMain)
  target_compile_definitions(${FB_TEST_TARGET} PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  add_test(NAME ${FB_TEST_TARGET} COMMAND ${FB_TEST_TARGET})
  if(NOT FB_TEST_LABELS)
    set(FB_TEST_LABELS imported unit)
  endif()
  set_property(TEST ${FB_TEST_TARGET} PROPERTY LABELS ${FB_TEST_LABELS})
endfunction()
