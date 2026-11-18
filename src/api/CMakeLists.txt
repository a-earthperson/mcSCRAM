# Leaf API types used by ir::pc and FTP/AIGER IO. This is not the
# role/provider/registry/pipeline framework from the May-2026 reset.

add_library(fuzzybits_api INTERFACE)
target_include_directories(fuzzybits_api INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

function(fuzzybits_ensure_catch2)
  if(TARGET Catch2::Catch2)
    return()
  endif()

  include(FetchContent)
  set(CATCH_INSTALL_DOCS OFF CACHE BOOL "Disable Catch2 docs" FORCE)
  set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "Disable Catch2 extras install" FORCE)
  FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(Catch2)
endfunction()

function(fuzzybits_add_imported_test)
  set(one_value_args TARGET)
  set(multi_value_args SOURCES LINK_LIBRARIES)
  cmake_parse_arguments(PARSE_ARGV 0 FB_TEST "" "${one_value_args}" "${multi_value_args}")

  add_executable(${FB_TEST_TARGET} ${FB_TEST_SOURCES})
  target_link_libraries(${FB_TEST_TARGET} PRIVATE ${FB_TEST_LINK_LIBRARIES} Catch2::Catch2WithMain)
  target_compile_definitions(${FB_TEST_TARGET} PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  add_test(NAME ${FB_TEST_TARGET} COMMAND ${FB_TEST_TARGET})
endfunction()
