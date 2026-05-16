#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "api/diagnostics.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"
#include "ir/pc/target.h"

namespace fuzzybits::ir::pc {

// Finalization is fallible because validation is part of sealing the semantic
// contract: malformed graphs produce diagnostics instead of partially valid IR.
struct finalize_result final {
    circuit output;
    api::diagnostic_list diagnostics;

    [[nodiscard]] auto ok() const -> bool { return !diagnostics.has_errors() && output.valid(); }
};

// Builder is the only mutable construction surface for R1. It accumulates
// columnar scratch data, validates semantic invariants, then seals into an
// immutable circuit.
class builder final {
  public:
    builder() = default;

    void reserve_nodes(std::uint32_t count) {
        storage_.op_codes.reserve(count);
        storage_.operand_offsets.reserve(count);
        storage_.operand_counts.reserve(count);
        storage_.probability_source_refs.reserve(count);
        storage_.cardinality_low.reserve(count);
        storage_.cardinality_high.reserve(count);
        storage_.constant_values.reserve(count);
    }

    [[nodiscard]] auto add_literal_probability(double value) -> probability_source_id {
        return add_probability_source(probability_source::literal_probability(value));
    }

    [[nodiscard]] auto add_parameter_probability(std::string_view name) -> probability_source_id {
        return add_probability_source(probability_source::parameter_probability(std::string(name)));
    }

    [[nodiscard]] auto add_unsupported_probability(std::string reason) -> probability_source_id {
        return add_probability_source(probability_source::unsupported_probability(std::move(reason)));
    }

    [[nodiscard]] auto add_basic_event(probability_source_id source) -> bool_node_id {
        return add_node(op_code::basic_event, std::span<edge_ref const>{}, source);
    }

    [[nodiscard]] auto add_constant(bool state) -> bool_node_id {
        return add_node(op_code::constant, std::span<edge_ref const>{}, probability_source_id::invalid(), 0, 0, state);
    }

    [[nodiscard]] auto add_house_event(bool state) -> bool_node_id {
        return add_node(op_code::house_event, std::span<edge_ref const>{}, probability_source_id::invalid(), 0, 0,
                        state);
    }

    [[nodiscard]] auto add_gate(op_code code, std::span<edge_ref const> operands) -> bool_node_id {
        return add_node(code, operands);
    }

    // Span is the canonical operand API: it accepts vectors, arrays, and other
    // contiguous storage without implying ownership. This initializer-list
    // overload exists only for small hand-built circuits and immediately
    // delegates to the span path, where operands are copied into builder storage.
    [[nodiscard]] auto add_gate(op_code code, std::initializer_list<edge_ref> operands) -> bool_node_id {
        return add_gate(code, std::span<edge_ref const>{operands.begin(), operands.size()});
    }

    [[nodiscard]] auto add_not(edge_ref operand) -> bool_node_id {
        const edge_ref operands[] = {operand};
        return add_gate(op_code::not_gate, operands);
    }

    [[nodiscard]] auto add_and(std::span<edge_ref const> operands) -> bool_node_id {
        return add_gate(op_code::and_gate, operands);
    }

    [[nodiscard]] auto add_and(std::initializer_list<edge_ref> operands) -> bool_node_id {
        return add_and(std::span<edge_ref const>{operands.begin(), operands.size()});
    }

    [[nodiscard]] auto add_or(std::span<edge_ref const> operands) -> bool_node_id {
        return add_gate(op_code::or_gate, operands);
    }

    [[nodiscard]] auto add_or(std::initializer_list<edge_ref> operands) -> bool_node_id {
        return add_or(std::span<edge_ref const>{operands.begin(), operands.size()});
    }

    [[nodiscard]] auto add_cardinality(std::uint32_t low, std::uint32_t high,
                                       std::span<edge_ref const> operands) -> bool_node_id {
        return add_node(op_code::cardinality, operands, probability_source_id::invalid(), low, high);
    }

    // Cardinality keeps the same overload shape as generic gates: span is the
    // storage-neutral form, initializer-list is a convenience for short tests and
    // import adapters that construct small operand lists inline.
    [[nodiscard]] auto add_cardinality(std::uint32_t low, std::uint32_t high,
                                       std::initializer_list<edge_ref> operands) -> bool_node_id {
        return add_cardinality(low, high, std::span<edge_ref const>{operands.begin(), operands.size()});
    }

    [[nodiscard]] auto add_target(bool_node_id node, observable_role role = observable_role::primary) -> target_id {
        const auto id = target_id(static_cast<target_id::value_type>(storage_.targets.size()));
        storage_.targets.push_back(target_view{
            .id = id,
            .node = node,
            .role = role,
        });
        return id;
    }

    // Seal performs semantic validation and moves the scratch columns into
    // shared read-only storage. It intentionally does not impose topological or
    // kind-partitioned layout; those are derived views in later phases.
    [[nodiscard]] auto seal() && -> finalize_result {
        auto diagnostics = validate();
        if (diagnostics.has_errors()) {
            return finalize_result{
                .output = {},
                .diagnostics = std::move(diagnostics),
            };
        }

        auto storage = std::make_shared<detail::circuit_storage>(std::move(storage_));
        auto shape = summarize(*storage);
        return finalize_result{
            .output = circuit(std::move(storage), std::move(shape)),
            .diagnostics = std::move(diagnostics),
        };
    }

  private:
    [[nodiscard]] auto add_probability_source(probability_source source) -> probability_source_id {
        const auto id =
            probability_source_id(static_cast<probability_source_id::value_type>(storage_.probability_sources.size()));
        storage_.probability_sources.push_back(std::move(source));
        return id;
    }

    [[nodiscard]] auto add_node(op_code code, std::span<edge_ref const> operands,
                                probability_source_id probability = probability_source_id::invalid(),
                                std::uint32_t cardinality_low = 0, std::uint32_t cardinality_high = 0,
                                bool constant_value = false) -> bool_node_id {
        const auto id = bool_node_id(static_cast<bool_node_id::value_type>(storage_.op_codes.size()));
        storage_.op_codes.push_back(code);
        storage_.operand_offsets.push_back(static_cast<std::uint32_t>(storage_.edges.size()));
        storage_.operand_counts.push_back(static_cast<std::uint32_t>(operands.size()));
        storage_.edges.insert(storage_.edges.end(), operands.begin(), operands.end());
        storage_.probability_source_refs.push_back(probability);
        storage_.cardinality_low.push_back(cardinality_low);
        storage_.cardinality_high.push_back(cardinality_high);
        storage_.constant_values.push_back(constant_value ? 1U : 0U);
        return id;
    }

    [[nodiscard]] auto validate() const -> api::diagnostic_list {
        api::diagnostic_list diagnostics;
        const auto node_count = storage_.op_codes.size();
        if (node_count == 0) {
            diagnostics.push(api::make_error(api::diagnostic_code("pc.empty-circuit"),
                                             "probabilistic circuit must contain at least one node"));
            return diagnostics;
        }

        bool endpoints_valid = true;
        for (std::uint32_t index = 0; index < node_count; ++index) {
            validate_node(index, diagnostics, endpoints_valid);
        }
        validate_probability_sources(diagnostics);
        validate_targets(diagnostics);
        if (endpoints_valid) {
            validate_acyclic(diagnostics);
        }
        return diagnostics;
    }

    void validate_node(std::uint32_t index, api::diagnostic_list &diagnostics, bool &endpoints_valid) const {
        const auto code = storage_.op_codes[index];
        const auto count = storage_.operand_counts[index];
        const auto offset = storage_.operand_offsets[index];

        if (is_leaf(code) && count != 0) {
            diagnostics.push(api::make_error(api::diagnostic_code("pc.leaf-operands"),
                                             "leaf node " + std::to_string(index) + " must not have operands"));
        }
        if (code == op_code::basic_event && !valid_probability_source(storage_.probability_source_refs[index])) {
            diagnostics.push(api::make_error(api::diagnostic_code("pc.invalid-probability-source"),
                                             "basic event " + std::to_string(index) +
                                                 " references an invalid probability source"));
        }
        if (code == op_code::not_gate && count != 1) {
            diagnostics.push(api::make_error(api::diagnostic_code("pc.invalid-arity"),
                                             "not gate " + std::to_string(index) + " must have exactly one operand"));
        }
        if (is_boolean_gate(code) && code != op_code::not_gate && count == 0) {
            diagnostics.push(api::make_error(api::diagnostic_code("pc.invalid-arity"),
                                             "gate " + std::to_string(index) + " must have at least one operand"));
        }
        if (code == op_code::cardinality) {
            validate_cardinality(index, diagnostics);
        }

        for (std::uint32_t operand_index = 0; operand_index < count; ++operand_index) {
            const edge_ref edge = storage_.edges[offset + operand_index];
            if (!valid_node(edge.source)) {
                endpoints_valid = false;
                diagnostics.push(api::make_error(api::diagnostic_code("pc.invalid-edge"),
                                                 "node " + std::to_string(index) +
                                                     " references an invalid operand"));
            }
        }
    }

    void validate_cardinality(std::uint32_t index, api::diagnostic_list &diagnostics) const {
        const auto count = storage_.operand_counts[index];
        const auto low = storage_.cardinality_low[index];
        const auto high = storage_.cardinality_high[index];
        if (count == 0) {
            diagnostics.push(api::make_error(api::diagnostic_code("pc.invalid-cardinality"),
                                             "cardinality node " + std::to_string(index) +
                                                 " must have at least one operand"));
            return;
        }
        if (low > high || high > count) {
            diagnostics.push(api::make_error(api::diagnostic_code("pc.invalid-cardinality"),
                                             "cardinality node " + std::to_string(index) +
                                                 " has bounds outside its fan-in"));
        }
    }

    void validate_probability_sources(api::diagnostic_list &diagnostics) const {
        for (std::uint32_t index = 0; index < storage_.probability_sources.size(); ++index) {
            auto const &source = storage_.probability_sources[index];
            if (source.kind == probability_source_kind::literal &&
                (!std::isfinite(source.literal) || source.literal < 0.0 || source.literal > 1.0)) {
                diagnostics.push(api::make_error(api::diagnostic_code("pc.invalid-literal-probability"),
                                                 "literal probability source " + std::to_string(index) +
                                                     " is outside [0, 1]"));
            }
            if (source.kind == probability_source_kind::parameter && source.parameter_name.empty()) {
                diagnostics.push(api::make_error(api::diagnostic_code("pc.invalid-parameter-probability"),
                                                 "parameter probability source " + std::to_string(index) +
                                                     " must have a name"));
            }
        }
    }

    void validate_targets(api::diagnostic_list &diagnostics) const {
        for (target_view const &target : storage_.targets) {
            if (!valid_node(target.node)) {
                diagnostics.push(api::make_error(api::diagnostic_code("pc.invalid-target"),
                                                 "target " + std::to_string(target.id.raw()) +
                                                     " references an invalid node"));
            }
        }
    }

    void validate_acyclic(api::diagnostic_list &diagnostics) const {
        // Kahn traversal over the operand relation validates DAG semantics
        // without requiring node ids to already be topologically sorted.
        const auto node_count = static_cast<std::uint32_t>(storage_.op_codes.size());
        std::vector<std::uint32_t> remaining_dependencies(node_count, 0);
        std::vector<std::uint32_t> dependent_counts(node_count, 0);

        for (std::uint32_t node = 0; node < node_count; ++node) {
            remaining_dependencies[node] = storage_.operand_counts[node];
            const auto offset = storage_.operand_offsets[node];
            const auto count = storage_.operand_counts[node];
            for (std::uint32_t operand = 0; operand < count; ++operand) {
                ++dependent_counts[storage_.edges[offset + operand].source.raw()];
            }
        }

        std::vector<std::uint32_t> dependent_offsets(node_count + 1, 0);
        for (std::uint32_t node = 0; node < node_count; ++node) {
            dependent_offsets[node + 1] = dependent_offsets[node] + dependent_counts[node];
        }

        std::vector<std::uint32_t> write_offsets = dependent_offsets;
        std::vector<std::uint32_t> dependents(dependent_offsets.back(), 0);
        for (std::uint32_t node = 0; node < node_count; ++node) {
            const auto offset = storage_.operand_offsets[node];
            const auto count = storage_.operand_counts[node];
            for (std::uint32_t operand = 0; operand < count; ++operand) {
                const auto source = storage_.edges[offset + operand].source.raw();
                dependents[write_offsets[source]++] = node;
            }
        }

        std::vector<std::uint32_t> ready;
        ready.reserve(node_count);
        for (std::uint32_t node = 0; node < node_count; ++node) {
            if (remaining_dependencies[node] == 0) {
                ready.push_back(node);
            }
        }

        for (std::uint32_t cursor = 0; cursor < ready.size(); ++cursor) {
            const auto source = ready[cursor];
            for (std::uint32_t index = dependent_offsets[source]; index < dependent_offsets[source + 1]; ++index) {
                const auto dependent = dependents[index];
                --remaining_dependencies[dependent];
                if (remaining_dependencies[dependent] == 0) {
                    ready.push_back(dependent);
                }
            }
        }

        if (ready.size() != node_count) {
            diagnostics.push(api::make_error(api::diagnostic_code("pc.cycle"),
                                             "probabilistic circuit contains a Boolean dependency cycle"));
        }
    }

    [[nodiscard]] auto valid_node(bool_node_id node) const noexcept -> bool {
        return node.valid() && node.raw() < storage_.op_codes.size();
    }

    [[nodiscard]] auto valid_probability_source(probability_source_id source) const noexcept -> bool {
        return source.valid() && source.raw() < storage_.probability_sources.size();
    }

    [[nodiscard]] static auto summarize(detail::circuit_storage const &storage) -> shape_summary {
        shape_summary shape;
        shape.node_count = storage.op_codes.size();
        shape.edge_count = storage.edges.size();
        shape.acyclic = true;
        shape.has_csr_edge_view = true;

        for (op_code code : storage.op_codes) {
            add_unique(shape.op_codes_present, code);
        }
        for (probability_source const &source : storage.probability_sources) {
            add_unique(shape.probability_source_kinds_present, source.kind);
            if (source.kind == probability_source_kind::unsupported) {
                shape.has_unsupported_probability_sources = true;
            }
        }
        for (target_view const &target : storage.targets) {
            add_unique(shape.observable_roles_present, target.role);
        }
        return shape;
    }

    template <typename value_type> static void add_unique(std::vector<value_type> &values, value_type value) {
        if (std::ranges::find(values, value) == values.end()) {
            values.push_back(value);
        }
    }

    detail::circuit_storage storage_;
};

} // namespace fuzzybits::ir::pc
