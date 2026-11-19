# Bundle AdaptiveCpp runtime next to the Python extension so the wheel is
# self-contained. AdaptiveCpp discovers:
#   dirname(libacpp-rt)/hipSYCL            — backend plugins
#   dirname(libacpp-common)/hipSYCL/ext    — opt/llc, libdevice.10.bc
#   dirname(libacpp-common)/hipSYCL/bitcode
# so the tree MUST sit in fuzzybits/lib/ (libacpp-*.so at that root).
#
# Only used when SKBUILD=ON (pip/scikit-build-core). Regular cmake --install
# keeps the system AdaptiveCpp prefix.

if(NOT SKBUILD)
  return()
endif()

find_program(ACPP_EXECUTABLE NAMES acpp REQUIRED)
get_filename_component(_acpp_bin_dir "${ACPP_EXECUTABLE}" DIRECTORY)
get_filename_component(_acpp_prefix "${_acpp_bin_dir}" DIRECTORY)

set(_fuzzybits_bundle "${CMAKE_CURRENT_LIST_DIR}/bundle_acpp.py")
set(_fuzzybits_relocate "${CMAKE_CURRENT_LIST_DIR}/relocate_wheel_rpath.py")

# FetchContent libxml2 is a shared lib (libxml2.so.16 / libxml2.dll) that is
# not on the system loader path. Drop it next to libacpp-rt / acpp-rt.dll.
if(TARGET LibXml2::LibXml2)
  install(CODE "
    set(_xml \"$<TARGET_FILE:LibXml2::LibXml2>\")
    set(_dest \"\${CMAKE_INSTALL_PREFIX}/fuzzybits/lib\")
    file(MAKE_DIRECTORY \"\${_dest}\")
    message(STATUS \"Bundling libxml2 \${_xml} -> \${_dest}\")
    file(COPY \"\${_xml}\" DESTINATION \"\${_dest}\")
    get_filename_component(_xml_dir \"\${_xml}\" DIRECTORY)
    file(GLOB _xml_sibs \"\${_xml_dir}/libxml2.so*\" \"\${_xml_dir}/libxml2*.dll\" \"\${_xml_dir}/*xml2*.dll\")
    foreach(_s \${_xml_sibs})
      file(COPY \"\${_s}\" DESTINATION \"\${_dest}\")
    endforeach()
  " COMPONENT scram)
endif()

install(CODE "
  set(_dest \"\${CMAKE_INSTALL_PREFIX}/fuzzybits/lib\")
  file(MAKE_DIRECTORY \"\${_dest}\")
  message(STATUS \"Bundling AdaptiveCpp runtime from ${_acpp_prefix} -> \${_dest}\")
  execute_process(
    COMMAND \"${Python3_EXECUTABLE}\" \"${_fuzzybits_bundle}\" \"${_acpp_prefix}\" \"\${_dest}\"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR \"bundle_acpp.py failed (rc=\${_rc}):\n\${_out}\n\${_err}\")
  endif()
  message(STATUS \"\${_out}\")
  if(NOT WIN32)
    execute_process(
      COMMAND \"${Python3_EXECUTABLE}\" \"${_fuzzybits_relocate}\" \"\${CMAKE_INSTALL_PREFIX}/fuzzybits\"
      RESULT_VARIABLE _rc
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR \"relocate_wheel_rpath.py failed (rc=\${_rc}):\n\${_out}\n\${_err}\")
    endif()
    message(STATUS \"\${_out}\")
  endif()
" COMPONENT scram)
