add_library(fuzzybits_io INTERFACE)
target_link_libraries(fuzzybits_io INTERFACE fuzzybits_api fuzzybits_ir)

add_subdirectory(ftp)
add_subdirectory(aiger)
