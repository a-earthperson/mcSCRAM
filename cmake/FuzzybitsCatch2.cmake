# Single Catch2 pin for the whole tree. ir::pc tests (#58) introduced v3.4.0;
# every other FetchContent of Catch2 must reuse this helper so CMake does not
# fetch the package twice (and so we do not drift to a second version).
include_guard(GLOBAL)

macro(fuzzybits_ensure_catch2)
  if(NOT TARGET Catch2::Catch2)
    include(FetchContent)
    set(CATCH_INSTALL_DOCS OFF CACHE BOOL "Disable Catch2 docs" FORCE)
    set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "Disable Catch2 extras install" FORCE)
    FetchContent_Declare(
      Catch2
      URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(Catch2)
  endif()
  if(DEFINED catch2_SOURCE_DIR)
    list(PREPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  endif()
endmacro()
