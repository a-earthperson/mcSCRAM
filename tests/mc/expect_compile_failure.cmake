# Builds TARGET in BUILD_DIR and succeeds iff the build fails, at least one
# compiler error matches EXPECT, and no compiler error fails to match it.
#
#   cmake -DBUILD_DIR=<dir> -DTARGET=<target> -DCONFIG=<cfg> -DEXPECT=<regex> -P expect_compile_failure.cmake

foreach(_var BUILD_DIR TARGET EXPECT)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR "${_var} is required")
  endif()
endforeach()

set(_config_args)
if(CONFIG)
  set(_config_args --config "${CONFIG}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --target "${TARGET}" ${_config_args}
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _out
)

if(_rc EQUAL 0)
  message(FATAL_ERROR "${TARGET} compiled, expected a failure matching: ${EXPECT}")
endif()

# Semicolons would split diagnostics when treated as a CMake list.
string(REPLACE ";" "," _out_flat "${_out}")
string(REGEX MATCHALL "[^\n]*error: [^\n]*" _errors "${_out_flat}")
set(_matched 0)
set(_unexpected)
foreach(_e IN LISTS _errors)
  if(_e MATCHES "${EXPECT}")
    math(EXPR _matched "${_matched} + 1")
  else()
    list(APPEND _unexpected "${_e}")
  endif()
endforeach()

if(_matched EQUAL 0 OR _unexpected)
  list(JOIN _unexpected "\n" _unexpected_text)
  message(FATAL_ERROR
    "${TARGET}: ${_matched} error(s) matched '${EXPECT}'; unexpected errors:\n${_unexpected_text}\n--- build output ---\n${_out}")
endif()

message(STATUS "${TARGET}: failed as expected (${_matched} matching error(s))")
