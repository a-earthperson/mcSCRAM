set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Non-wheel installs keep the historical module name so existing Docker images
# and PYTHONPATH=/usr/local/lib/python3.12/dist-packages keep working.
# Wheels ship one distinctly named extension per variant so generic and
# cuda-sm70 can coexist on sys.path (see python/fuzzybits/_runtime.py).
if(SKBUILD)
  if(ACPP_TARGETS MATCHES "cuda:sm_70")
    set(FUZZYBITS_NATIVE_MODULE _native_cuda_sm70)
  else()
    set(FUZZYBITS_NATIVE_MODULE _native_generic)
  endif()
  set(MCSCRAM_PYTHON_INSTALL_DIR "fuzzybits" CACHE PATH
    "Directory where the fuzzybits native Python extension is installed")
else()
  set(FUZZYBITS_NATIVE_MODULE mcscram_native)
  set(MCSCRAM_PYTHON_INSTALL_DIR "${Python3_SITEARCH}" CACHE PATH
    "Directory where the mcscram_native Python extension is installed")
endif()

pybind11_add_module(${FUZZYBITS_NATIVE_MODULE} MODULE bindings.cpp)

find_package(_adaptivecpp)
add_sycl_to_target(TARGET ${FUZZYBITS_NATIVE_MODULE} SOURCES bindings.cpp)

target_compile_definitions(${FUZZYBITS_NATIVE_MODULE}
  PRIVATE
    FUZZYBITS_NATIVE_MODULE=${FUZZYBITS_NATIVE_MODULE}
    FUZZYBITS_NATIVE_MODULE_NAME="${FUZZYBITS_NATIVE_MODULE}"
)

target_link_libraries(${FUZZYBITS_NATIVE_MODULE}
  PRIVATE
    scram
    mc
    ${LIBS}
)

target_include_directories(${FUZZYBITS_NATIVE_MODULE}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/targets/scram
)

if(SKBUILD)
  set_target_properties(${FUZZYBITS_NATIVE_MODULE} PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH_USE_LINK_PATH FALSE
  )
endif()

install(
  TARGETS ${FUZZYBITS_NATIVE_MODULE}
  LIBRARY DESTINATION "${MCSCRAM_PYTHON_INSTALL_DIR}"
  COMPONENT scram
)

if(SKBUILD)
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/python/fuzzybits/"
    DESTINATION fuzzybits
    COMPONENT scram
    FILES_MATCHING PATTERN "*.py"
  )
  install(
    FILES "${PROJECT_SOURCE_DIR}/python/mcscram_native.py"
    DESTINATION "."
    COMPONENT scram
  )
  include("${PROJECT_SOURCE_DIR}/cmake/BundleAdaptiveCpp.cmake")
endif()
