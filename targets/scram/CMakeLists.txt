add_subdirectory(utils)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

####################### Begin SCRAM CLI config ##################### {{{

set(SCRAM_CLI_SOURCES
        main.cpp
)

# Define a variable for the installation base directory
set(INSTALL_BASE_DIR "$ENV{HOME}/.local")

if (NOT TARGET scram-cli)
    add_executable(scram-cli ${SCRAM_CLI_SOURCES})
endif ()

target_link_libraries(scram-cli scram-cli-utils scram canopy)

# Install the executable to ~/.local/bin
install(
        TARGETS scram-cli
        RUNTIME DESTINATION ${INSTALL_BASE_DIR}/bin
        COMPONENT scram-cli
)

####################### End SCRAM CLI config ##################### }}}
