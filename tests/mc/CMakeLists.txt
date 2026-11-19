# Monte Carlo tests. Label `mc`. Every MC invocation has TIMEOUT 60 so a
# non-terminating sampler (#22) fails fast. Device selection is via
# ACPP_VISIBILITY_MASK (test presets: omp on CPU, cuda/metal on GPU).

if(NOT FUZZYBITS_WITH_MC)
  return()
endif()

# add_sycl_to_target needs AdaptiveCpp::acpp-rt in this directory scope;
# src/mc's find_package is not visible to the tests/ tree.
find_package(_adaptivecpp)

set(MC_TEST_TIMEOUT 60)

add_executable(mc_cli_correctness_tests cli_correctness_tests.cc)
# Catch2WithMain must precede mc: libmc PUBLIC-links ${LIBS}, which still
# contains Boost::unit_test_framework and would otherwise own main.
target_link_libraries(mc_cli_correctness_tests PRIVATE Catch2::Catch2WithMain ${SCRAM_TEST_LIBS} scram mc)
target_compile_definitions(mc_cli_correctness_tests PRIVATE
  SCRAM_CLI_PATH="$<TARGET_FILE:scram-cli>"
  PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
target_include_directories(mc_cli_correctness_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tests"
)
add_dependencies(mc_cli_correctness_tests scram-cli)

foreach(_model chinese das9201 isp9601 baobab2 small_tree bscu)
  add_test(NAME mc.cli.${_model} COMMAND mc_cli_correctness_tests "[${_model}]")
  set_tests_properties(mc.cli.${_model} PROPERTIES
    LABELS "mc"
    TIMEOUT ${MC_TEST_TIMEOUT}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
endforeach()

add_executable(mc_fzb_cli_identity_tests fzb_cli_identity_tests.cc)
target_link_libraries(mc_fzb_cli_identity_tests PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(mc_fzb_cli_identity_tests PRIVATE
  SCRAM_CLI_PATH="$<TARGET_FILE:scram-cli>"
  FZB_PATH="$<TARGET_FILE:fzb>"
  PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
add_dependencies(mc_fzb_cli_identity_tests scram-cli fzb)
foreach(_case n65536 n2p24 seed smalltree bscu)
  add_test(NAME mc.fzb_cli_identity.${_case} COMMAND mc_fzb_cli_identity_tests "[${_case}]")
  set_tests_properties(mc.fzb_cli_identity.${_case} PROPERTIES
    LABELS "mc"
    TIMEOUT ${MC_TEST_TIMEOUT}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
endforeach()

add_executable(mc_bitpack_identity_tests bitpack_identity_tests.cc)
target_link_libraries(mc_bitpack_identity_tests PRIVATE Catch2::Catch2WithMain mc)
target_include_directories(mc_bitpack_identity_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tests"
)
add_sycl_to_target(TARGET mc_bitpack_identity_tests SOURCES bitpack_identity_tests.cc)
add_test(NAME mc.bitpack_identity COMMAND mc_bitpack_identity_tests)
set_tests_properties(mc.bitpack_identity PROPERTIES
  LABELS "mc"
  TIMEOUT ${MC_TEST_TIMEOUT}
)

add_executable(mc_device_selection_tests device_selection_tests.cc)
target_link_libraries(mc_device_selection_tests PRIVATE Catch2::Catch2WithMain mc)
target_include_directories(mc_device_selection_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tests"
)
add_sycl_to_target(TARGET mc_device_selection_tests SOURCES device_selection_tests.cc)
add_test(NAME mc.device_selection COMMAND mc_device_selection_tests)
set_tests_properties(mc.device_selection PROPERTIES
  LABELS "mc"
  TIMEOUT ${MC_TEST_TIMEOUT}
)

# Removal canaries for src/mc/compat/metal.h. Not labelled `mc`. Default
# presets exclude `metal-canary`. macos-metal runs them with WILL_FAIL.
add_executable(mc_metal_canary_tests metal_canary_tests.cc)
target_link_libraries(mc_metal_canary_tests PRIVATE Catch2::Catch2WithMain mc)
target_include_directories(mc_metal_canary_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tests"
)
add_sycl_to_target(TARGET mc_metal_canary_tests SOURCES metal_canary_tests.cc)
add_test(NAME mc.metal_canary.reduce64 COMMAND mc_metal_canary_tests "[reduce64]")
add_test(NAME mc.metal_canary.double COMMAND mc_metal_canary_tests "[double]")
set_tests_properties(mc.metal_canary.reduce64 mc.metal_canary.double PROPERTIES
  LABELS "metal-canary"
  TIMEOUT ${MC_TEST_TIMEOUT}
)
if(APPLE)
  set_tests_properties(mc.metal_canary.reduce64 mc.metal_canary.double PROPERTIES
    WILL_FAIL TRUE
  )
endif()
