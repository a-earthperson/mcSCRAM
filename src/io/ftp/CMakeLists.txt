add_library(fuzzybits_io_ftp STATIC
  read.cpp
  write.cpp
  detail/parser.cpp
  detail/target_selection.cpp
  detail/builder.cpp
  detail/writer.cpp
  descriptor.h
  detail/model.h
)
set_target_properties(fuzzybits_io_ftp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fuzzybits_io_ftp PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_link_libraries(fuzzybits_io_ftp
  PUBLIC fuzzybits_io
  PRIVATE fuzzybits_ir_pc
)

if(BUILD_TESTING)
  fuzzybits_ensure_catch2()
  fuzzybits_add_imported_test(
    TARGET fuzzybits_io_ftp_tests
    SOURCES tests/read_tests.cc tests/write_tests.cc
    LINK_LIBRARIES fuzzybits_io_ftp
  )
endif()
