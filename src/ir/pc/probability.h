#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace fuzzybits::ir::pc {

// R1 supports only probability sources that can be represented before the full
// expression evaluator RFC. Unsupported sources survive ingestion as diagnostics
// instead of being silently flattened or guessed.
enum class probability_source_kind : std::uint8_t {
    literal,
    parameter,
    unsupported,
};

// A probability source is the payload referenced by a basic event. Parameter and
// unsupported sources keep enough identity for planning-time compatibility
// checks; expression trees are deliberately not modeled here yet.
struct probability_source final {
    probability_source_kind kind = probability_source_kind::unsupported;
    double literal = 0.0;
    std::string parameter_name;
    std::string diagnostic;

    [[nodiscard]] static auto literal_probability(const double value) -> probability_source {
        return probability_source{
            .kind = probability_source_kind::literal,
            .literal = value,
        };
    }

    [[nodiscard]] static auto parameter_probability(std::string name) -> probability_source {
        return probability_source{
            .kind = probability_source_kind::parameter,
            .parameter_name = std::move(name),
        };
    }

    [[nodiscard]] static auto unsupported_probability(std::string reason) -> probability_source {
        return probability_source{
            .kind = probability_source_kind::unsupported,
            .diagnostic = std::move(reason),
        };
    }
};

[[nodiscard]] inline auto is_deterministic(probability_source const &source) noexcept -> bool {
    return source.kind == probability_source_kind::literal || source.kind == probability_source_kind::parameter;
}

} // namespace fuzzybits::ir::pc
