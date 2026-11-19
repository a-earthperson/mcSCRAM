# Wheel RPATH / install_name for SKBUILD installs.
# Linux: $ORIGIN so auditwheel + patchelf can relocate.
# Darwin: @loader_path so delocate can rewrite without flattening hipSYCL/.

if(APPLE)
  set(FUZZYBITS_WHEEL_INSTALL_RPATH "@loader_path;@loader_path/lib")
else()
  set(FUZZYBITS_WHEEL_INSTALL_RPATH "$ORIGIN;$ORIGIN/lib")
endif()

function(fuzzybits_set_wheel_rpath tgt)
  if(NOT SKBUILD)
    return()
  endif()
  if(NOT TARGET ${tgt})
    return()
  endif()
  set_target_properties(${tgt} PROPERTIES
    INSTALL_RPATH "${FUZZYBITS_WHEEL_INSTALL_RPATH}"
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH_USE_LINK_PATH FALSE
  )
endfunction()
