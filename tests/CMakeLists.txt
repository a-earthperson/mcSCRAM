# vim: set foldmarker={{{,}}} foldlevel=0 foldmethod=marker:
######################## Begin find Catch2 ###################### {{{
# Catch2 is fetched via FetchContent in the main CMakeLists.txt
# find_package is called in the main CMakeLists.txt before this subdirectory
#include(Catch)
######################## End find Catch2 ###################### }}}

######################## Begin SCRAM test config ###################### {{{
# Include the project headers.
include_directories("${CMAKE_SOURCE_DIR}/src")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

### Begin SCRAM test source list ### {{{
set(SCRAM_CORE_TEST_SOURCE
  version_tests.cc
  linear_map_tests.cc
  linear_set_tests.cc
  xml_stream_tests.cc
  settings_tests.cc
  project_tests.cc
  element_tests.cc
  event_tests.cc
  expression_tests.cc
  extern_function_tests.cc
  ccf_group_tests.cc
  fault_tree_tests.cc
  alignment_tests.cc
  pdag_tests.cc
  initializer_tests.cc
  serialization_tests.cc
  risk_analysis_tests.cc
  bench_core_tests.cc
  bench_two_train_tests.cc
  bench_lift_tests.cc
  bench_ne574_tests.cc
  bench_theatre_tests.cc
  bench_three_motor_tests.cc
  bench_200_event_tests.cc
  bench_small_tree_tests.cc
  bench_bscu_tests.cc
  bench_chinese_tree_tests.cc
  bench_baobab1_tests.cc
  bench_baobab2_tests.cc
  bench_CEA9601_tests.cc
  bench_hipps_tests.cc
  bench_attack.cc
  bench_gas_leak.cc
  performance_tests.cc
  )
### End SCRAM test source list ### }}}

# Make tests switch and run in the test directory.
configure_file(scram_unit_test_driver.cc.in scram_unit_test_driver.cc @ONLY)

add_executable(scram_tests ${CMAKE_CURRENT_BINARY_DIR}/scram_unit_test_driver.cc ${SCRAM_CORE_TEST_SOURCE})
target_link_libraries(scram_tests ${LIBS} scram Catch2::Catch2WithMain)
target_compile_options(scram_tests PRIVATE $<$<CONFIG:DEBUG>:${SCRAM_CXX_FLAGS_DEBUG}>)

install(
  TARGETS scram_tests
  RUNTIME DESTINATION bin
  COMPONENT testing
  )
######################## End SCRAM test config ###################### }}}

######################## Begin Dummy DLL config ###################### {{{
# Create dummy library to test dynamic loading and extern function.
add_library(scram_dummy_extern SHARED scram_dummy_extern.cc)
target_compile_options(scram_dummy_extern PRIVATE $<$<CONFIG:DEBUG>:${SCRAM_CXX_FLAGS_DEBUG}>)
if(WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "")
install(
  TARGETS scram_dummy_extern
  LIBRARY DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
  ARCHIVE DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/static"
  RUNTIME DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
  COMPONENT testing
  )
endif()
######################## End Dummy DLL config ###################### }}}

# Add the Philox PRNG test files to the existing test suite

# Philox Statistical Tests
add_executable(philox_statistical_tests
    philox_statistical_tests.cc
    ${CMAKE_SOURCE_DIR}/src/mc/kernel/basic_event.h
    ${CMAKE_SOURCE_DIR}/src/mc/event/node.h
)

target_link_libraries(philox_statistical_tests
    Catch2::Catch2WithMain
    ${SYCL_LIBRARIES}
)

target_include_directories(philox_statistical_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

# Philox Benchmark Tests
add_executable(philox_benchmark_tests
    philox_benchmark_tests.cc
    ${CMAKE_SOURCE_DIR}/src/mc/kernel/basic_event.h
    ${CMAKE_SOURCE_DIR}/src/mc/event/node.h
)

target_link_libraries(philox_benchmark_tests
    Catch2::Catch2WithMain
    ${SYCL_LIBRARIES}
)

target_include_directories(philox_benchmark_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

# Philox Seeding Tests
add_executable(philox_seeding_tests
    philox_seeding_tests.cc
    ${CMAKE_SOURCE_DIR}/src/mc/kernel/basic_event.h
    ${CMAKE_SOURCE_DIR}/src/mc/event/node.h
)

target_link_libraries(philox_seeding_tests
    Catch2::Catch2WithMain
    ${SYCL_LIBRARIES}
)

target_include_directories(philox_seeding_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

# Add tests to CTest using Catch2 discovery
include(Catch)

# Discover tests for each executable
catch_discover_tests(philox_statistical_tests
    TEST_PREFIX "philox_statistical_"
    PROPERTIES TIMEOUT 300
)

catch_discover_tests(philox_benchmark_tests
    TEST_PREFIX "philox_benchmark_"
    PROPERTIES TIMEOUT 300
)

catch_discover_tests(philox_seeding_tests
    TEST_PREFIX "philox_seeding_"
    PROPERTIES TIMEOUT 300
)
