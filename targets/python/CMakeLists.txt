set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(MCSCRAM_PYTHON_INSTALL_DIR "${Python3_SITEARCH}" CACHE PATH
  "Directory where the mcscram_native Python extension is installed")

pybind11_add_module(mcscram_native MODULE bindings.cpp)

find_package(_adaptivecpp)
add_sycl_to_target(TARGET mcscram_native SOURCES bindings.cpp)

target_link_libraries(mcscram_native
  PRIVATE
    scram
    mc
    ${LIBS}
)

target_include_directories(mcscram_native
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/targets/scram
)

install(
  TARGETS mcscram_native
  LIBRARY DESTINATION "${MCSCRAM_PYTHON_INSTALL_DIR}"
  COMPONENT scram
)
