#pragma once

#include "api/ids.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace fuzzybits::api {

using option_value = std::variant<std::monostate, bool, std::int64_t, double, std::string>;

struct option_spec final {
    option_key key;
    std::string name;
    option_value default_value;
    std::string help;
};

struct option_schema final {
    std::vector<option_spec> entries;
};

class option_set final {
  public:
    void set(option_key key, option_value value) { values_.insert_or_assign(std::move(key), std::move(value)); }

    template <typename value_type> [[nodiscard]] auto get(option_key const &key) const -> std::optional<value_type> {
        const auto found = values_.find(key);
        if (found == values_.end()) {
            return std::nullopt;
        }
        if (auto const *value = std::get_if<value_type>(&found->second)) {
            return *value;
        }
        return std::nullopt;
    }

  private:
    std::unordered_map<option_key, option_value> values_;
};

} // namespace fuzzybits::api
