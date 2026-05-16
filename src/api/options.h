#pragma once

#include "api/diagnostics.h"
#include "api/identifiers.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace fuzzybits::api {

    enum class option_value_kind : std::uint8_t {
        none,
        boolean,
        integer,
        real,
        text,
        text_list,
    };

    using option_value =
        std::variant<std::monostate, bool, std::int64_t, double, std::string, std::vector<std::string>>;

    struct option_constraint final {
        diagnostic_code code;
        std::string message;
        std::vector<option_key> required_options;
        std::vector<option_key> conflicts_with;
    };

    struct option_spec final {
        option_key key;
        option_value_kind kind = option_value_kind::none;
        option_value default_value;
        bool required = false;
        std::string help;
        std::string owner;
        std::vector<option_constraint> constraints;
    };

    class option_schema final {
    public:
        using storage_type = std::vector<option_spec>;
        using const_iterator = storage_type::const_iterator;

        option_schema() = default;

        option_schema(std::string owner, storage_type options)
            : owner_(std::move(owner)), options_(std::move(options)) {}

        [[nodiscard]] auto owner() const noexcept -> std::string const & {
            return owner_;
        }

        [[nodiscard]] auto options() const noexcept -> storage_type const & {
            return options_;
        }

        [[nodiscard]] auto find(option_key const &key) const noexcept -> option_spec const * {
            for (auto const &option : options_) {
                if (option.key == key) {
                    return &option;
                }
            }
            return nullptr;
        }

        [[nodiscard]] auto begin() const noexcept -> const_iterator {
            return options_.begin();
        }

        [[nodiscard]] auto end() const noexcept -> const_iterator {
            return options_.end();
        }

    private:
        std::string owner_;
        storage_type options_;
    };

    class option_set final {
    public:
        using storage_type = std::unordered_map<option_key, option_value>;

        auto set(option_key key, option_value value) -> option_set & {
            values_.insert_or_assign(std::move(key), std::move(value));
            return *this;
        }

        [[nodiscard]] auto contains(option_key const &key) const -> bool {
            return values_.contains(key);
        }

        [[nodiscard]] auto find(option_key const &key) const -> option_value const * {
            auto const iter = values_.find(key);
            return iter == values_.end() ? nullptr : &iter->second;
        }

        template <typename value_type>
        [[nodiscard]] auto get(option_key const &key) const -> std::optional<value_type> {
            auto const *value = find(key);
            if (value == nullptr) {
                return std::nullopt;
            }

            auto const *typed = std::get_if<value_type>(value);
            if (typed == nullptr) {
                return std::nullopt;
            }

            return *typed;
        }

        [[nodiscard]] auto empty() const noexcept -> bool {
            return values_.empty();
        }

        [[nodiscard]] auto size() const noexcept -> std::size_t {
            return values_.size();
        }

        [[nodiscard]] auto raw() const noexcept -> storage_type const & {
            return values_;
        }

    private:
        storage_type values_;
    };

    struct resolved_options final {
        option_schema schema;
        option_set values;
        diagnostic_list diagnostics;

        [[nodiscard]] auto ok() const -> bool {
            return !diagnostics.has_errors();
        }
    };

} // namespace fuzzybits::api
