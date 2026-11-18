set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(fzb main.cpp)
target_link_libraries(fzb PRIVATE fuzzybits_compile Boost::program_options)
# libscram.so references DirectEval symbols that live in SYCL libmc (layering
# inversion; proper fix is out of scope for Phase 1, see #44). Host stubs
# satisfy the loader only when libmc is not linked. Do not compile them into
# fzb when FUZZYBITS_WITH_MC=ON: ELF interposition would bind libscram.so to
# the stubs and fzb run would throw / observe nothing.
if(NOT APPLE AND NOT FUZZYBITS_WITH_MC)
  target_sources(fzb PRIVATE "${PROJECT_SOURCE_DIR}/src/compile/direct_eval_host_stubs.cc")
  target_include_directories(fzb PRIVATE "${PROJECT_SOURCE_DIR}/src/mc")
elseif(FUZZYBITS_WITH_MC)
  target_link_libraries(fzb PRIVATE mc)
endif()
install(
  TARGETS fzb
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT fzb
)
