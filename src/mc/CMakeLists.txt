set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(MC_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/analysis.cc
)

# Optionally collect public headers for installation
set(MC_HEADERS
        event/node.h
        working_set.h
        kernel/basic_event.h
        kernel/gate.h
        kernel/tally.h
        queue/scheduler.h
        queue/layer_manager.h
        queue/layer_manager.tpp
        queue/kernel_builder.h
        queue/queueable.h
)

find_package(_adaptivecpp)
# Add AdaptiveCpp libraries to the global LIBS list
list(APPEND LIBS AdaptiveCpp::acpp-rt)

# Build mc as shared library explicitly (dependencies are static)
add_library(mc SHARED ${MC_SRC} ${MC_HEADERS})

# Mark header files as SYSTEM headers to suppress warnings from third-party code
set_target_properties(mc PROPERTIES PUBLIC_HEADER "${MC_HEADERS}")

# Make sure the compiler can find the mc headers when the library is consumed
target_include_directories(mc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/mc>
)

# Link against the core SCRAM library, system libs, and the AdaptiveCpp SYCL target.
# The imported SYCL::SYCL target is provided by AdaptiveCpp when `find_package(_adaptivecpp)`
# is invoked from the top-level CMakeLists.txt.
# `${LIBS}` comes from the parent scope and contains third-party dependencies.

target_link_libraries(mc
  PUBLIC
    scram
    ${LIBS}
)

# Enable AdaptiveCpp device compilation for the library sources
add_sycl_to_target(TARGET mc SOURCES ${MC_SRC})

# Propagate the same debug compile options used by the rest of SCRAM
# so that the new target benefits from the strict checks in Debug builds.

target_compile_options(mc PRIVATE $<$<CONFIG:DEBUG>:${SCRAM_CXX_FLAGS_DEBUG}>)

# Inherit the include directories defined at higher levels (src/ and project root).
# Additional includes specific to this sub-module can be added here if needed.

# Install the library next to the main SCRAM library so that downstream
# consumers can find it in the same prefix.
install(
  TARGETS mc
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mc
  COMPONENT scram
)