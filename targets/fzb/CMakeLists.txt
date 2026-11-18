set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(fzb main.cpp)
target_link_libraries(fzb PRIVATE fuzzybits_compile Boost::program_options)
# libscram.so references DirectEval symbols that live in SYCL libmc. Provide
# host stubs so fzb does not need AdaptiveCpp. Apple already compiles them
# into libscram.
if(NOT APPLE)
  target_sources(fzb PRIVATE "${PROJECT_SOURCE_DIR}/src/compile/direct_eval_host_stubs.cc")
  target_include_directories(fzb PRIVATE "${PROJECT_SOURCE_DIR}/src/mc")
endif()
install(
  TARGETS fzb
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT fzb
)
