#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pcfb.h"

namespace scram::core {
class Pdag;
class Settings;
}  // namespace scram::core

namespace fuzzybits::compile {

/// Effective preprocessor flags, matching `scram-cli --pdag -c N [--no-kn …]`
/// and `run_compile`.
struct Options {
  int compilation_level = 2;
  bool expand_atleast_gates = false;  // --no-kn
  bool expand_xor_gates = false;      // --no-xor
  bool recognize_atleast_groups = true;
  bool recognize_xor_groups = true;
  bool keep_null_gates = false;
  bool ccf = false;
  bool allow_extern = false;
  std::optional<std::string> target;

  /// Reconstruct the effective CLI fragment written to `compile.options`.
  [[nodiscard]] std::string to_cli_string() const;
};

struct Compiled {
  std::vector<std::byte> bytes;
  std::string model_name;
  std::string target_id;
  std::string target_name;
};

/// Apply `options` to a Settings object (compilation flags only).
void apply_options(scram::core::Settings &settings, const Options &options);

/// OpenPSA/MEF XML → preprocessed PDAG → pcfb core-1.1 buffer.
[[nodiscard]] pcfb::Result<Compiled> compile_files(const std::vector<std::string> &input_files,
                                                   const Options &options);

/// Lower an already-preprocessed PDAG. `event_id` of each SAMPLE is the PDAG index.
[[nodiscard]] pcfb::Result<std::vector<std::byte>>
pdag_to_pcfb(scram::core::Pdag &pdag, const Options &options, std::string_view model_name,
             std::string_view query_name, std::span<const std::byte> source_sha256);

}  // namespace fuzzybits::compile
