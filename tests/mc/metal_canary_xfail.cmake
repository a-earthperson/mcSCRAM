# Invert a Metal removal-canary exit status (xfail-strict).
#
# AdaptiveCpp aborts on unsupported device IR (SIGABRT). CTest classifies that
# as an Exception, so WILL_FAIL does not treat it as an expected failure.
# This wrapper turns any non-zero child status into success. A clean pass
# (rv == 0) fails the build: AdaptiveCpp now lowers the construct and the
# matching shim in src/mc/compat/metal.h must be deleted.
if(NOT CANARY_COMMAND)
  message(FATAL_ERROR "CANARY_COMMAND is required")
endif()
if(NOT CANARY_FILTER)
  message(FATAL_ERROR "CANARY_FILTER is required")
endif()
execute_process(COMMAND "${CANARY_COMMAND}" "${CANARY_FILTER}" RESULT_VARIABLE rv)
if(rv EQUAL 0)
  message(FATAL_ERROR
    "metal canary '${CANARY_FILTER}' passed unexpectedly; remove the matching shim in src/mc/compat/metal.h")
endif()
