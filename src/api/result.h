#pragma once

#include "api/diagnostics.h"

#include <optional>
#include <utility>

namespace fuzzybits::api {

template <typename value_type> class result final {
  public:
    result() = default;

    [[nodiscard]] static auto ok(value_type value, diagnostic_list diagnostics = {}) -> result {
        result out;
        out.value_ = std::move(value);
        out.diagnostics_ = std::move(diagnostics);
        return out;
    }

    [[nodiscard]] static auto fail(diagnostic diagnostic) -> result {
        diagnostic_list diagnostics;
        diagnostics.push(std::move(diagnostic));
        return fail(std::move(diagnostics));
    }

    [[nodiscard]] static auto fail(diagnostic_list diagnostics) -> result {
        result out;
        out.diagnostics_ = std::move(diagnostics);
        return out;
    }

    [[nodiscard]] auto has_value() const noexcept -> bool { return value_.has_value(); }
    [[nodiscard]] explicit operator bool() const noexcept { return has_value() && !diagnostics_.has_errors(); }

    [[nodiscard]] auto value() & -> value_type & { return *value_; }
    [[nodiscard]] auto value() const & -> value_type const & { return *value_; }
    [[nodiscard]] auto value() && -> value_type { return std::move(*value_); }

    [[nodiscard]] auto diagnostics() const noexcept -> diagnostic_list const & { return diagnostics_; }
    [[nodiscard]] auto diagnostics() noexcept -> diagnostic_list & { return diagnostics_; }

  private:
    std::optional<value_type> value_;
    diagnostic_list diagnostics_;
};

} // namespace fuzzybits::api
