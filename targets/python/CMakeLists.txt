set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Wheels (and non-wheel builds) ship one distinctly named extension per variant
# so generic and cuda-smNN can coexist on sys.path (python/fuzzybits/_runtime.py).
if(ACPP_TARGETS MATCHES "^cuda:sm_([0-9]+)$")
  set(FUZZYBITS_NATIVE_MODULE _native_cuda_sm${CMAKE_MATCH_1})
else()
  set(FUZZYBITS_NATIVE_MODULE _native_generic)
endif()
if(SKBUILD)
  set(MCSCRAM_PYTHON_INSTALL_DIR "fuzzybits" CACHE PATH
    "Directory where the fuzzybits native Python extension is installed")
else()
  set(MCSCRAM_PYTHON_INSTALL_DIR "${Python3_SITEARCH}" CACHE PATH
    "Directory where the fuzzybits native Python extension is installed")
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
    fuzzybits_compile
    ${LIBS}
)

target_include_directories(${FUZZYBITS_NATIVE_MODULE}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/targets/scram
)

if(SKBUILD)
  fuzzybits_set_wheel_rpath(${FUZZYBITS_NATIVE_MODULE})
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
  include("${PROJECT_SOURCE_DIR}/cmake/BundleAdaptiveCpp.cmake")
endif()
