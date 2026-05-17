#pragma once

#include "api/ids.h"

#include <algorithm>
#include <initializer_list>
#include <utility>
#include <vector>

namespace fuzzybits::api {

    class capability_set final {
    public:
        capability_set() = default;
        capability_set(const std::initializer_list<capability_id> capabilities) {
            for (auto capability : capabilities) {
                add(std::move(capability));
            }
        }

        void add(capability_id capability) {
            if (!contains(capability)) {
                values_.push_back(std::move(capability));
            }
        }

        [[nodiscard]] auto contains(capability_id const &capability) const -> bool {
            return std::ranges::find(values_, capability) != values_.end();
        }

        [[nodiscard]] auto contains_all(capability_set const &other) const -> bool {
            for (auto const &capability : other.values_) {
                if (!contains(capability)) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] auto missing_from(capability_set const &available) const -> capability_set {
            capability_set missing;
            for (auto const &capability : values_) {
                if (!available.contains(capability)) {
                    missing.add(capability);
                }
            }
            return missing;
        }

        [[nodiscard]] auto empty() const noexcept -> bool {
            return values_.empty();
        }
        [[nodiscard]] auto values() const noexcept -> std::vector<capability_id> const & {
            return values_;
        }

    private:
        std::vector<capability_id> values_;
    };

} // namespace fuzzybits::api
