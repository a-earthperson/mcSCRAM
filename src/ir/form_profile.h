#pragma once

#include "api/ids.h"
#include "ir/descriptor.h"
#include "ir/forms.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::ir {

    namespace detail {

        enum class form_node : std::uint8_t {
            bool_dag,
            probabilistic_boolean_dag,
            nnf,
            xag,
            flat_nnf,
            dnnf,
            deterministic_nnf,
            aig,
            rnf,
            fprm,
            cnf,
            dnf,
            smooth_dnnf,
            deterministic_dnnf,
            pi,
            implicant_cover,
            pprm,
            epi,
            essential_implicant,
            smooth_deterministic_dnnf,
            bdd,
            free_bdd,
            obdd,
            sdd,
            robdd,
            count,
        };

        using form_mask = std::uint64_t;

        struct form_entry final {
            form_node node;
            api::form_id (*id)();
            std::array<form_node, 2> parents;
            std::uint8_t parent_count = 0;
        };

        [[nodiscard]] constexpr auto form_node_index(const form_node node) -> std::size_t {
            return static_cast<std::size_t>(node);
        }

        [[nodiscard]] constexpr auto form_node_bit(const form_node node) -> form_mask {
            return form_mask{1} << form_node_index(node);
        }

        [[nodiscard]] inline auto form_entries() -> std::span<form_entry const> {
            static constexpr std::array entries{
                form_entry{form_node::bool_dag, forms::bool_dag, {}, 0},
                form_entry{form_node::probabilistic_boolean_dag, forms::probabilistic_boolean_dag, {form_node::bool_dag}, 1},
                form_entry{form_node::nnf, forms::nnf, {form_node::bool_dag}, 1},
                form_entry{form_node::xag, forms::xag, {form_node::bool_dag}, 1},
                form_entry{form_node::flat_nnf, forms::flat_nnf, {form_node::nnf}, 1},
                form_entry{form_node::dnnf, forms::dnnf, {form_node::nnf}, 1},
                form_entry{form_node::deterministic_nnf, forms::deterministic_nnf, {form_node::nnf}, 1},
                form_entry{form_node::aig, forms::aig, {form_node::xag}, 1},
                form_entry{form_node::rnf, forms::rnf, {form_node::xag}, 1},
                form_entry{form_node::fprm, forms::fprm, {form_node::rnf}, 1},
                form_entry{form_node::cnf, forms::cnf, {form_node::flat_nnf}, 1},
                form_entry{form_node::dnf, forms::dnf, {form_node::flat_nnf, form_node::dnnf}, 2},
                form_entry{form_node::smooth_dnnf, forms::smooth_dnnf, {form_node::dnnf}, 1},
                form_entry{form_node::deterministic_dnnf, forms::deterministic_dnnf, {form_node::dnnf, form_node::deterministic_nnf}, 2},
                form_entry{form_node::pi, forms::pi, {form_node::cnf}, 1},
                form_entry{form_node::implicant_cover, forms::implicant_cover, {form_node::dnf}, 1},
                form_entry{form_node::pprm, forms::pprm, {form_node::fprm}, 1},
                form_entry{form_node::epi, forms::epi, {form_node::pi}, 1},
                form_entry{form_node::essential_implicant, forms::essential_implicant, {form_node::implicant_cover}, 1},
                form_entry{form_node::smooth_deterministic_dnnf, forms::smooth_deterministic_dnnf, {form_node::smooth_dnnf, form_node::deterministic_dnnf}, 2},
                form_entry{form_node::bdd, forms::bdd, {form_node::deterministic_dnnf}, 1},
                form_entry{form_node::free_bdd, forms::free_bdd, {form_node::bdd}, 1},
                form_entry{form_node::obdd, forms::obdd, {form_node::free_bdd}, 1},
                form_entry{form_node::sdd, forms::sdd, {form_node::smooth_deterministic_dnnf}, 1},
                form_entry{form_node::robdd, forms::robdd, {form_node::obdd}, 1},
            };
            return entries;
        }

        [[nodiscard]] inline auto known_form_node(api::form_id const &form) -> form_node {
            for (form_entry const &entry : form_entries()) {
                if (entry.id() == form) {
                    return entry.node;
                }
            }
            return form_node::count;
        }

        [[nodiscard]] inline auto known_form_masks() -> std::array<form_mask, form_node_index(form_node::count)> const & {
            static const auto masks = [] {
                std::array<form_mask, form_node_index(form_node::count)> result{};
                for (form_entry const &entry : form_entries()) {
                    auto mask = form_node_bit(entry.node);
                    for (std::uint8_t index = 0; index < entry.parent_count; ++index) {
                        mask |= result[form_node_index(entry.parents[index])];
                    }
                    result[form_node_index(entry.node)] = mask;
                }
                return result;
            }();
            return masks;
        }

        [[nodiscard]] inline auto hierarchy_mask_for(api::form_id const &form) -> form_mask {
            const auto node = known_form_node(form);
            if (node == form_node::count) {
                return 0;
            }
            return known_form_masks()[form_node_index(node)];
        }

    } // namespace detail

    class form_profile final {
    public:
        form_profile() = default;

        [[nodiscard]] static auto from_exact_form(api::form_id form) -> form_profile {
            form_profile profile;
            profile.exact_form_ = std::move(form);
            profile.add_satisfied_form(profile.exact_form_);
            return profile;
        }

        [[nodiscard]] static auto from_satisfied_forms(std::span<api::form_id const> forms) -> form_profile {
            form_profile profile;
            for (api::form_id const &form : forms) {
                profile.add_satisfied_form(form);
            }
            return profile;
        }

        void add_satisfied_form(api::form_id form) {
            if (const auto mask = detail::hierarchy_mask_for(form); mask != 0) {
                mask_ |= mask;
                return;
            }
            if (std::ranges::find(extra_forms_, form) == extra_forms_.end()) {
                extra_forms_.push_back(std::move(form));
            }
        }

        [[nodiscard]] auto exact_form() const noexcept -> api::form_id const & {
            return exact_form_;
        }

        [[nodiscard]] auto satisfies(api::form_id const &form) const -> bool {
            if (const auto node = detail::known_form_node(form); node != detail::form_node::count) {
                return satisfies(node);
            }
            return std::ranges::find(extra_forms_, form) != extra_forms_.end();
        }

        [[nodiscard]] auto satisfied_form_ids() const -> std::vector<api::form_id> {
            std::vector<api::form_id> forms;
            forms.reserve(detail::form_entries().size() + extra_forms_.size());
            for (detail::form_entry const &entry : detail::form_entries()) {
                if (satisfies(entry.node)) {
                    forms.push_back(entry.id());
                }
            }
            forms.insert(forms.end(), extra_forms_.begin(), extra_forms_.end());
            return forms;
        }

        [[nodiscard]] auto satisfied_form_strings() const -> std::vector<std::string> {
            auto ids = satisfied_form_ids();
            std::vector<std::string> strings;
            strings.reserve(ids.size());
            for (api::form_id const &id : ids) {
                strings.push_back(id.str());
            }
            return strings;
        }

        [[nodiscard]] auto is_bool_dag() const noexcept -> bool {
            return satisfies(detail::form_node::bool_dag);
        }

        [[nodiscard]] auto is_probabilistic_boolean_dag() const noexcept -> bool {
            return satisfies(detail::form_node::probabilistic_boolean_dag);
        }

        [[nodiscard]] auto is_nnf() const noexcept -> bool {
            return satisfies(detail::form_node::nnf);
        }

        [[nodiscard]] auto is_xag() const noexcept -> bool {
            return satisfies(detail::form_node::xag);
        }

        [[nodiscard]] auto is_aig() const noexcept -> bool {
            return satisfies(detail::form_node::aig);
        }

        [[nodiscard]] auto is_cnf() const noexcept -> bool {
            return satisfies(detail::form_node::cnf);
        }

        [[nodiscard]] auto is_dnf() const noexcept -> bool {
            return satisfies(detail::form_node::dnf);
        }

        [[nodiscard]] auto is_bdd() const noexcept -> bool {
            return satisfies(detail::form_node::bdd);
        }

        [[nodiscard]] auto is_free_bdd() const noexcept -> bool {
            return satisfies(detail::form_node::free_bdd);
        }

        [[nodiscard]] auto is_obdd() const noexcept -> bool {
            return satisfies(detail::form_node::obdd);
        }

        [[nodiscard]] auto is_robdd() const noexcept -> bool {
            return satisfies(detail::form_node::robdd);
        }

    private:
        [[nodiscard]] auto satisfies(detail::form_node const node) const noexcept -> bool {
            return (mask_ & detail::form_node_bit(node)) != 0;
        }

        api::form_id exact_form_;
        detail::form_mask mask_ = 0;
        std::vector<api::form_id> extra_forms_;
    };

    [[nodiscard]] inline auto form_profile_for(descriptor const &descriptor) -> form_profile {
        auto profile = form_profile::from_exact_form(descriptor.form);
        for (api::form_id const &form : descriptor.satisfied_forms) {
            profile.add_satisfied_form(form);
        }
        return profile;
    }

    [[nodiscard]] inline auto descriptor_satisfies_form(descriptor const &descriptor, api::form_id const &form) -> bool {
        if (descriptor.form == form || std::ranges::find(descriptor.satisfied_forms, form) != descriptor.satisfied_forms.end()) {
            return true;
        }
        return form_profile_for(descriptor).satisfies(form);
    }

} // namespace fuzzybits::ir
