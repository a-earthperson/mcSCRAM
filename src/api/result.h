#pragma once

#include "api/diagnostics.h"

#include <type_traits>
#include <utility>
#include <variant>

namespace fuzzybits::api {

    template <typename value_type> class result final {
    public:
        static_assert(!std::is_reference_v<value_type>);
        static_assert(!std::is_void_v<value_type>);

        [[nodiscard]] static auto ok(value_type value) -> result {
            return result(std::move(value));
        }

        [[nodiscard]] static auto fail(diagnostic_list diagnostics) -> result {
            return result(std::move(diagnostics));
        }

        [[nodiscard]] static auto fail(diagnostic diagnostic_item) -> result {
            return result(diagnostic_list(std::move(diagnostic_item)));
        }

        [[nodiscard]] auto has_value() const noexcept -> bool {
            return std::holds_alternative<value_type>(state_);
        }

        [[nodiscard]] explicit operator bool() const noexcept {
            return has_value();
        }

        [[nodiscard]] auto value() const & -> value_type const & {
            return std::get<value_type>(state_);
        }

        [[nodiscard]] auto value() & -> value_type & {
            return std::get<value_type>(state_);
        }

        [[nodiscard]] auto value() && -> value_type && {
            return std::move(std::get<value_type>(state_));
        }

        [[nodiscard]] auto diagnostics() const & -> diagnostic_list const & {
            if (auto const *diagnostics = std::get_if<diagnostic_list>(&state_)) {
                return *diagnostics;
            }

            static diagnostic_list const empty;
            return empty;
        }

    private:
        explicit result(value_type value) : state_(std::move(value)) {}

        explicit result(diagnostic_list diagnostics) : state_(std::move(diagnostics)) {}

        std::variant<value_type, diagnostic_list> state_;
    };

    template <> class result<void> final {
    public:
        [[nodiscard]] static auto ok() -> result {
            return {true, {}};
        }

        [[nodiscard]] static auto fail(diagnostic_list diagnostics) -> result {
            return {false, std::move(diagnostics)};
        }

        [[nodiscard]] static auto fail(diagnostic diagnostic_item) -> result {
            return {false, diagnostic_list(std::move(diagnostic_item))};
        }

        [[nodiscard]] auto has_value() const noexcept -> bool {
            return ok_;
        }

        [[nodiscard]] explicit operator bool() const noexcept {
            return has_value();
        }

        [[nodiscard]] auto diagnostics() const & -> diagnostic_list const & {
            return diagnostics_;
        }

    private:
        result(const bool ok, diagnostic_list diagnostics) : ok_(ok), diagnostics_(std::move(diagnostics)) {}

        bool ok_ = false;
        diagnostic_list diagnostics_;
    };

} // namespace fuzzybits::api
