#pragma once

#include <compare>
#include <cstdint>
#include <limits>

namespace fuzzybits::ir::pc {

// Strongly typed dense ids prevent accidental cross-table indexing while still
// compiling down to a single integer in storage and hot traversal paths.
template <typename tag_type> class id final {
  public:
    using value_type = std::uint32_t;

    static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();

    constexpr id() = default;

    explicit constexpr id(const value_type value) noexcept : value_(value) {}

    [[nodiscard]] static constexpr auto invalid() noexcept -> id { return id(); }

    [[nodiscard]] constexpr auto valid() const noexcept -> bool { return value_ != invalid_value; }

    [[nodiscard]] constexpr auto raw() const noexcept -> value_type { return value_; }

    friend auto operator<=>(id const &, id const &) = default;

  private:
    value_type value_ = invalid_value;
};

namespace id_tags {
struct bool_node;
struct probability_source;
struct target;
} // namespace id_tags

// These ids are storage-local. Transforms that remap nodes are responsible for
// returning provenance maps rather than promising global id stability.
using bool_node_id = id<id_tags::bool_node>;
using probability_source_id = id<id_tags::probability_source>;
using target_id = id<id_tags::target>;

} // namespace fuzzybits::ir::pc
