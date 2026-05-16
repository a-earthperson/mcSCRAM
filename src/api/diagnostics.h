#pragma once

#include "api/identifiers.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::api {

    enum class diagnostic_severity : std::uint8_t {
        note,
        warning,
        error,
        fatal,
    };

    struct source_location final {
        std::string file;
        std::uint64_t line = 0;
        std::uint64_t column = 0;

        [[nodiscard]] auto known() const noexcept -> bool {
            return !file.empty() || line != 0 || column != 0;
        }
    };

    struct diagnostic final {
        diagnostic_severity severity = diagnostic_severity::error;
        diagnostic_code code;
        std::string message;
        source_location location;
        std::vector<std::string> notes;
    };

    class diagnostic_list final {
    public:
        using storage_type = std::vector<diagnostic>;
        using iterator = storage_type::iterator;
        using const_iterator = storage_type::const_iterator;

        diagnostic_list() = default;

        explicit diagnostic_list(diagnostic first) {
            diagnostics_.push_back(std::move(first));
        }

        auto push(diagnostic item) -> diagnostic_list & {
            diagnostics_.push_back(std::move(item));
            return *this;
        }

        auto append(diagnostic_list other) -> diagnostic_list & {
            diagnostics_.insert(diagnostics_.end(), std::make_move_iterator(other.diagnostics_.begin()),
                                std::make_move_iterator(other.diagnostics_.end()));
            return *this;
        }

        [[nodiscard]] auto empty() const noexcept -> bool {
            return diagnostics_.empty();
        }

        [[nodiscard]] auto size() const noexcept -> std::size_t {
            return diagnostics_.size();
        }

        [[nodiscard]] auto has_errors() const -> bool {
            return std::ranges::any_of(diagnostics_, [](diagnostic const &item) {
                return item.severity == diagnostic_severity::error || item.severity == diagnostic_severity::fatal;
            });
        }

        [[nodiscard]] auto raw() const noexcept -> storage_type const & {
            return diagnostics_;
        }

        [[nodiscard]] auto begin() noexcept -> iterator {
            return diagnostics_.begin();
        }

        [[nodiscard]] auto end() noexcept -> iterator {
            return diagnostics_.end();
        }

        [[nodiscard]] auto begin() const noexcept -> const_iterator {
            return diagnostics_.begin();
        }

        [[nodiscard]] auto end() const noexcept -> const_iterator {
            return diagnostics_.end();
        }

    private:
        storage_type diagnostics_;
    };

    [[nodiscard]] inline auto make_error(diagnostic_code code, std::string message) -> diagnostic {
        return diagnostic{
            .severity = diagnostic_severity::error,
            .code = std::move(code),
            .message = std::move(message),
        };
    }

} // namespace fuzzybits::api
