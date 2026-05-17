#pragma once

#include <compare>
#include <functional>
#include <string>
#include <utility>

namespace fuzzybits::api {

class id final {
  public:
    id() = default;
    explicit id(std::string value) : value_(std::move(value)) {}

    [[nodiscard]] auto empty() const noexcept -> bool { return value_.empty(); }
    [[nodiscard]] auto str() const noexcept -> std::string const & { return value_; }

    friend auto operator<=>(id const &, id const &) = default;

  private:
    std::string value_;
};

using provider_id = id;
using endpoint_id = id;
using type_id = id;
using form_id = id;
using format_id = id;
using capability_id = id;
using option_key = id;
using result_id = id;
using target_id = id;
using stage_id = id;

} // namespace fuzzybits::api

namespace std {

template <> struct hash<fuzzybits::api::id> {
    [[nodiscard]] auto operator()(fuzzybits::api::id const &value) const noexcept -> std::size_t {
        return std::hash<std::string>{}(value.str());
    }
};

} // namespace std
