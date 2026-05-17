#pragma once

#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::api {

using diagnostic_code = std::string;

enum class diagnostic_severity {
    note,
    warning,
    error,
};

struct diagnostic final {
    diagnostic_severity severity = diagnostic_severity::note;
    std::string code;
    std::string message;
};

class diagnostic_list final {
  public:
    void push(diagnostic item) { items_.push_back(std::move(item)); }

    [[nodiscard]] auto empty() const noexcept -> bool { return items_.empty(); }
    [[nodiscard]] auto size() const noexcept -> std::size_t { return items_.size(); }
    [[nodiscard]] auto items() const noexcept -> std::vector<diagnostic> const & { return items_; }

    [[nodiscard]] auto has_errors() const noexcept -> bool {
        for (auto const &item : items_) {
            if (item.severity == diagnostic_severity::error) {
                return true;
            }
        }
        return false;
    }

    void append(diagnostic_list other) {
        items_.insert(items_.end(), std::make_move_iterator(other.items_.begin()),
                      std::make_move_iterator(other.items_.end()));
    }

  private:
    std::vector<diagnostic> items_;
};

[[nodiscard]] inline auto make_error(std::string code, std::string message) -> diagnostic {
    return diagnostic{.severity = diagnostic_severity::error, .code = std::move(code), .message = std::move(message)};
}

[[nodiscard]] inline auto make_warning(std::string code, std::string message) -> diagnostic {
    return diagnostic{.severity = diagnostic_severity::warning, .code = std::move(code), .message = std::move(message)};
}

[[nodiscard]] inline auto make_note(std::string code, std::string message) -> diagnostic {
    return diagnostic{.severity = diagnostic_severity::note, .code = std::move(code), .message = std::move(message)};
}

} // namespace fuzzybits::api
