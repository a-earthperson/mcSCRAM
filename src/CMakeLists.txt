set(CMAKE_INCLUDE_CURRENT_DIR ON)

####################### Begin SCRAM core lib config ##################### {{{
### Begin SCRAM core source list ### {{{
set(SCRAM_CORE_SRC
  ext/version.cc
  ext/bimap.h
  env.cc
  logger.cc
  settings.cc
  xml.cc
  project.cc
  # Generated embedded schema files
  "${PROJECT_BINARY_DIR}/generated/input_schema.cpp"
  "${PROJECT_BINARY_DIR}/generated/project_schema.cpp"
  "${PROJECT_BINARY_DIR}/generated/report_schema.cpp"
  element.cc
  expression.cc
  parameter.cc
  expression/conditional.cc
  expression/constant.cc
  expression/numerical.cc
  expression/exponential.cc
  expression/random_deviate.cc
  expression/test_event.cc
  expression/extern.cc
  event.cc
  substitution.cc
  ccf_group.cc
  fault_tree.cc
  event_tree.cc
  alignment.cc
  model.cc
  cycle.cc
  pdag.cc
  preprocessor.cc
  mocus.cc
  bdd.cc
  zbdd.cc
  analysis.cc
  fault_tree_analysis.cc
  probability_analysis.cc
  importance_analysis.cc
  uncertainty_analysis.cc
  event_tree_analysis.cc
  reporter.cc
  serialization.cc
  initializer.cc
  risk_analysis.cc
  )
### End SCRAM core source list ### }}}

set(SCRAM_MC_SRC
  mc/core/preprocessor.cc
  mc/core/probability_analysis.cc
  mc/core/importance_analysis.cc
)

if(APPLE OR WIN32)
  # Darwin dylibs and Windows DLLs cannot leave DirectEval symbols undefined
  # in libscram and resolve them later from libmc. Build the DirectEval
  # implementation into scram so the shared library is self-contained and
  # there is no scram <-> mc shared-library cycle.
  find_package(_adaptivecpp)
  # Runtime SYCL entrypoint: AdaptiveCpp::acpp-rt is directory-scoped here, so
  # add_sycl_to_target from src/runtime/ cannot see it. pcfb is linked later.
  list(APPEND SCRAM_MC_SRC "${CMAKE_CURRENT_SOURCE_DIR}/runtime/run.cc")
  list(APPEND SCRAM_CORE_SRC ${SCRAM_MC_SRC})
  # Host-only MC statistics (Boost.Math) — not passed to add_sycl_to_target.
  list(APPEND SCRAM_CORE_SRC
    mc/scheduler/convergence_policy.cc
    mc/stats/info_gain.cc
    runtime/pdag_from_pcfb.cc
  )
endif()

# Linux: shared libscram.so. Darwin/Windows: static into the consumer
# (scram-cli, .pyd). Windows CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS does not
# export private static data (Logger::report_level_), so a scram.dll
# leaves scram-cli.exe with unresolved data symbols.
if(WIN32)
  add_library(scram STATIC ${SCRAM_CORE_SRC})
  set_target_properties(scram PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
  add_library(scram SHARED ${SCRAM_CORE_SRC})
endif()
target_link_libraries(scram PRIVATE ${LIBS})
target_compile_options(scram PRIVATE $<$<CONFIG:DEBUG>:${SCRAM_CXX_FLAGS_DEBUG}>)

if(APPLE OR WIN32)
  add_sycl_to_target(TARGET scram SOURCES ${SCRAM_MC_SRC})
  target_include_directories(scram PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/mc")
endif()

# Disable unused variable warnings for generated schema files
set_source_files_properties(
  "${PROJECT_BINARY_DIR}/generated/input_schema.cpp"
  "${PROJECT_BINARY_DIR}/generated/project_schema.cpp"
  "${PROJECT_BINARY_DIR}/generated/report_schema.cpp"
  PROPERTIES COMPILE_FLAGS "-Wno-unused-const-variable"
)

# Install the library to standard system directories respecting CMAKE_INSTALL_PREFIX.
# Wheels co-locate libscram next to the Python extension so RPATH $ORIGIN works.
# On Windows the DLL is the RUNTIME artefact; keep it next to the .pyd.
if(SKBUILD)
  set(_scram_lib_dest "fuzzybits")
  set(_scram_runtime_dest "fuzzybits")
else()
  set(_scram_lib_dest ${CMAKE_INSTALL_LIBDIR})
  set(_scram_runtime_dest ${CMAKE_INSTALL_BINDIR})
endif()
install(
        TARGETS scram
        COMPONENT scram
        LIBRARY DESTINATION ${_scram_lib_dest}
        ARCHIVE DESTINATION ${_scram_lib_dest}
        RUNTIME DESTINATION ${_scram_runtime_dest}
)
if(SKBUILD)
  set_target_properties(scram PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH_USE_LINK_PATH FALSE
  )
endif()

# add mc (SYCL). Host-only `fzb compile` sets FUZZYBITS_WITH_MC=OFF.
if(FUZZYBITS_WITH_MC)
  add_subdirectory(mc)
  # Make the SYCL mc library available to downstream targets
  # by appending it to the common ${LIBS} list that all executables use.
  list(APPEND LIBS mc)
else()
  add_library(mc INTERFACE)
  target_include_directories(mc INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mc>
  )
endif()

