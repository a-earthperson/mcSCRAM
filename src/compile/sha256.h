#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace fuzzybits::compile {

/// FIPS 180-4 SHA-256. Returns the 32-byte digest.
[[nodiscard]] std::array<std::uint8_t, 32> sha256(std::span<const std::byte> data);

[[nodiscard]] std::array<std::uint8_t, 32> sha256_file(const std::string &path);

/// Concatenate files in order and hash. Used for general.source_sha256.
[[nodiscard]] std::array<std::uint8_t, 32> sha256_files(const std::vector<std::string> &paths);

[[nodiscard]] std::string sha256_hex(std::span<const std::uint8_t, 32> digest);

}  // namespace fuzzybits::compile
