#pragma once

#include "compile/pcfb_writer.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fuzzybits::compile {

inline constexpr std::uint32_t kNombriloAnd = 0;
inline constexpr std::uint32_t kNombriloOr = 1;
inline constexpr std::uint32_t kNombriloXor = 0xFFFFFFFFu;

inline constexpr const char *kNombriloFormat = "nombrilo.circuit";
inline constexpr const char *kNombriloVersion = "1";
inline constexpr const char *kNombriloSourceFormat = "nombrilo.circuit-1";

/// In-memory nombrilo.circuit v1. `weights` absent means a variant file.
struct NombriloCircuit {
  std::uint32_t base_count = 0;
  std::vector<std::uint32_t> operators;
  std::vector<std::uint32_t> offsets;
  std::vector<std::uint32_t> children;
  std::vector<std::string> names;
  std::vector<std::uint32_t> name_lits;
  std::optional<std::vector<double>> weights;
  /// Full safetensors `__metadata__` map (string values), including core keys.
  std::map<std::string, std::string> metadata;
};

struct NombriloCompileOptions {
  std::optional<std::string> target;
  /// Explicit sibling weights file for variant circuits. Empty = auto-detect.
  std::optional<std::string> weights_file;
};

struct NombriloExportOptions {
  /// When true, store pcfb-only fields in `__metadata__` under `fuzzybits.pcfb.*`
  /// so nombrilo → pcfb can restore them. Strict plain nombrilo omits them.
  bool extras = true;
};

[[nodiscard]] bool looks_like_nombrilo(std::span<const std::byte> bytes);
[[nodiscard]] bool looks_like_nombrilo_file(const std::string &path);

[[nodiscard]] pcfb::Result<NombriloCircuit> read_nombrilo(std::span<const std::byte> bytes);
[[nodiscard]] pcfb::Result<NombriloCircuit> read_nombrilo_file(const std::string &path);

[[nodiscard]] pcfb::Result<std::vector<std::byte>> write_nombrilo(const NombriloCircuit &circuit);

/// Load a variant's missing `weights` from `weights_path` (or a sibling base
/// file next to `source_path`). Fails if `base_count` or input names differ.
[[nodiscard]] pcfb::Result<void> inherit_nombrilo_weights(NombriloCircuit &circuit,
                                                          const std::string &source_path,
                                                          const std::optional<std::string> &weights_path);

[[nodiscard]] pcfb::Result<Compiled> nombrilo_to_pcfb(const NombriloCircuit &circuit,
                                                      const NombriloCompileOptions &options,
                                                      std::span<const std::byte> source_sha256,
                                                      std::string_view model_name,
                                                      std::string_view source_path = {});

[[nodiscard]] pcfb::Result<NombriloCircuit> pcfb_to_nombrilo(const pcfb::ir::Circuit &circuit,
                                                             const NombriloExportOptions &options);

[[nodiscard]] pcfb::Result<Compiled> compile_nombrilo_file(const std::string &path,
                                                           const NombriloCompileOptions &options);

}  // namespace fuzzybits::compile
