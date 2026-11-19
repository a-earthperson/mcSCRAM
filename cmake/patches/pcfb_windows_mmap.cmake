# WIN32-only: rewrite pcfb's POSIX mmap I/O to MapViewOfFile.
# Applied after FetchContent so Linux/macOS keep upstream io.cpp unchanged.
# Fails configure if the expected POSIX snippets are gone (pin moved).

function(fuzzybits_pcfb_apply_windows_mmap src_dir)
  if(NOT WIN32)
    return()
  endif()
  set(_io "${src_dir}/src/io.cpp")
  if(NOT EXISTS "${_io}")
    message(FATAL_ERROR "pcfb Windows mmap patch: ${_io} not found")
  endif()
  file(READ "${_io}" _contents)
  string(REPLACE "\r\n" "\n" _contents "${_contents}")
  if(_contents MATCHES "CreateFileW")
    return()
  endif()
  set(_posix_inc "#include <fcntl.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n#include <unistd.h>")
  if(NOT _contents MATCHES "#include <sys/mman.h>")
    message(FATAL_ERROR
      "pcfb Windows mmap patch: ${_io} no longer includes <sys/mman.h>; update cmake/patches/pcfb_windows_mmap.cmake")
  endif()
  set(_win_inc [=[#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <string>]=])
  string(REPLACE "${_posix_inc}" "${_win_inc}" _contents "${_contents}")

  set(_posix_body [=[        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd == -1) {
            return std::unexpected(Error{Error::Code::IoError, std::string{"open failed: "}+ std::strerror(errno)});
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            return std::unexpected(Error{Error::Code::StatFailed, std::string{"fstat failed: "}+ std::strerror(errno)});
        }
        if (st.st_size == 0) {
            ::close(fd);
            return std::unexpected(Error{Error::Code::IoError, "file is empty"});
        }
        void *addr = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            ::close(fd);
            return std::unexpected(Error{Error::Code::MapFailed, std::string{"mmap failed: "}+ std::strerror(errno)});
        }
        // Build shared_ptr with custom deleter to own the mapping & fd
        auto deleter = [size=static_cast<size_t>(st.st_size), fd](const void *p){
            ::munmap(const_cast<void*>(p), size);
            ::close(fd);
        };
        std::shared_ptr<void const> owner{addr, deleter};
        *base_out  = reinterpret_cast<const std::byte*>(addr);
        *size_out  = static_cast<size_t>(st.st_size);
        return owner;]=])
  set(_win_body [=[        HANDLE file = CreateFileW(
            p.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            return std::unexpected(Error{Error::Code::IoError,
                "CreateFileW failed: " + std::to_string(GetLastError())});
        }
        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(file, &sz)) {
            CloseHandle(file);
            return std::unexpected(Error{Error::Code::StatFailed,
                "GetFileSizeEx failed: " + std::to_string(GetLastError())});
        }
        if (sz.QuadPart <= 0) {
            CloseHandle(file);
            return std::unexpected(Error{Error::Code::IoError, "file is empty"});
        }
        HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) {
            CloseHandle(file);
            return std::unexpected(Error{Error::Code::MapFailed,
                "CreateFileMappingW failed: " + std::to_string(GetLastError())});
        }
        void *addr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!addr) {
            CloseHandle(mapping);
            CloseHandle(file);
            return std::unexpected(Error{Error::Code::MapFailed,
                "MapViewOfFile failed: " + std::to_string(GetLastError())});
        }
        auto deleter = [file, mapping](const void *view) {
            UnmapViewOfFile(view);
            CloseHandle(mapping);
            CloseHandle(file);
        };
        std::shared_ptr<void const> owner{addr, deleter};
        *base_out  = reinterpret_cast<const std::byte*>(addr);
        *size_out  = static_cast<size_t>(sz.QuadPart);
        return owner;]=])
  if(NOT _contents MATCHES "int fd = ::open")
    message(FATAL_ERROR
      "pcfb Windows mmap patch: map_file body not found in ${_io}; update cmake/patches/pcfb_windows_mmap.cmake")
  endif()
  string(REPLACE "${_posix_body}" "${_win_body}" _contents "${_contents}")
  if(NOT _contents MATCHES "CreateFileW")
    message(FATAL_ERROR "pcfb Windows mmap patch: replacement did not apply to ${_io}")
  endif()
  file(WRITE "${_io}" "${_contents}")
  message(STATUS "pcfb: patched src/io.cpp mmap → MapViewOfFile (WIN32)")
endfunction()
