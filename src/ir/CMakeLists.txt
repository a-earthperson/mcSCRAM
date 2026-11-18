add_library(fuzzybits_ir INTERFACE)
target_link_libraries(fuzzybits_ir INTERFACE fuzzybits_api)

add_subdirectory(pc)
