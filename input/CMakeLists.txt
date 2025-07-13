# Copy all example input directories and their contents into the shared directory.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION share/scram/input
        COMPONENT examples
        FILES_MATCHING PATTERN "*.xml")
