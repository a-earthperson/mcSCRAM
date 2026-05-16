#pragma once

#include "api/diagnostics.h"
#include "api/identifiers.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::api {

    class feature_set final {
    public:
        using storage_type = std::vector<capability>;
        using const_iterator = storage_type::const_iterator;

        feature_set() = default;

        feature_set(const std::initializer_list<capability> features) {
            for (auto const &feature : features) {
                add(feature);
            }
        }

        auto add(capability feature) -> feature_set & {
            if (auto const position = std::ranges::lower_bound(features_, feature);
                position == features_.end() || *position != feature) {
                features_.insert(position, std::move(feature));
            }
            return *this;
        }

        [[nodiscard]] auto contains(capability const &feature) const -> bool {
            return std::ranges::binary_search(features_, feature);
        }

        [[nodiscard]] auto contains_all(feature_set const &required) const -> bool {
            return std::ranges::all_of(required.features_,
                                       [this](capability const &feature) { return contains(feature); });
        }

        [[nodiscard]] auto missing_from(feature_set const &available) const -> feature_set {
            feature_set missing;
            for (auto const &feature : features_) {
                if (!available.contains(feature)) {
                    missing.add(feature);
                }
            }
            return missing;
        }

        [[nodiscard]] auto empty() const noexcept -> bool {
            return features_.empty();
        }

        [[nodiscard]] auto size() const noexcept -> std::size_t {
            return features_.size();
        }

        [[nodiscard]] auto raw() const noexcept -> storage_type const & {
            return features_;
        }

        [[nodiscard]] auto begin() const noexcept -> const_iterator {
            return features_.begin();
        }

        [[nodiscard]] auto end() const noexcept -> const_iterator {
            return features_.end();
        }

    private:
        storage_type features_;
    };

    struct compatibility_report final {
        bool compatible = true;
        feature_set required;
        feature_set available;
        feature_set missing;
        diagnostic_list diagnostics;

        [[nodiscard]] auto ok() const -> bool {
            return compatible && missing.empty() && !diagnostics.has_errors();
        }
    };

    [[nodiscard]] inline auto
    check_features(feature_set required, feature_set available,
                   const diagnostic_code &missing_capability_code = diagnostic_code("missing-capability"))
        -> compatibility_report {
        auto missing = required.missing_from(available);
        diagnostic_list diagnostics;

        for (auto const &feature : missing) {
            diagnostics.push(make_error(missing_capability_code, "missing required capability: " + feature.str()));
        }

        return compatibility_report{
            .compatible = missing.empty(),
            .required = std::move(required),
            .available = std::move(available),
            .missing = std::move(missing),
            .diagnostics = std::move(diagnostics),
        };
    }

} // namespace fuzzybits::api
