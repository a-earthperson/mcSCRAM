#pragma once

#include "api/ir.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"
#include "ir/pc/target.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace fuzzybits::ir::pc {

// Structured shape data carries parameterized compatibility facts that do not
// fit in api::feature_set, such as present opcodes and probability source kinds.
struct shape_summary final {
    std::uint64_t node_count = 0;
    std::uint64_t edge_count = 0;
    std::uint32_t node_id_width = 32;
    bool acyclic = false;
    bool has_csr_edge_view = false;
    bool has_topological_view = false;
    bool has_kind_bucket_view = false;
    bool has_unsupported_probability_sources = false;
    // These are deterministic, insertion-ordered unique lists. They stay
    // vector-backed because descriptors are small and serialized/iterated more
    // often than searched.
    std::vector<op_code> op_codes_present;
    std::vector<probability_source_kind> probability_source_kinds_present;
    std::vector<observable_role> observable_roles_present;
};

// PC uses stable public ids so api::ir_handle can carry the concrete storage
// without requiring pipeline/compiler/solver public headers to expose it.
[[nodiscard]] inline auto provider_id() -> api::provider_id { return api::provider_id("fuzzybits.pc"); }

[[nodiscard]] inline auto backend_id() -> api::backend_id { return api::backend_id("fuzzybits.ir.pc"); }

[[nodiscard]] inline auto storage_type_id() -> api::type_id { return api::type_id("fuzzybits.ir.pc.circuit"); }

namespace capability {

// Capabilities remain coarse-grained booleans for planner filtering. Detailed
// constraints stay in shape_summary rather than encoded into capability names.
[[nodiscard]] inline auto acyclic() -> api::capability { return api::capability("pc.acyclic"); }

[[nodiscard]] inline auto has_csr_edge_view() -> api::capability {
    return api::capability("pc.has_csr_edge_view");
}

[[nodiscard]] inline auto has_topological_view() -> api::capability {
    return api::capability("pc.has_topological_view");
}

[[nodiscard]] inline auto has_kind_bucket_view() -> api::capability {
    return api::capability("pc.has_kind_bucket_view");
}

[[nodiscard]] inline auto has_literal_probability_sources() -> api::capability {
    return api::capability("pc.has_literal_probability_sources");
}

[[nodiscard]] inline auto has_parameter_probability_sources() -> api::capability {
    return api::capability("pc.has_parameter_probability_sources");
}

[[nodiscard]] inline auto has_unsupported_probability_sources() -> api::capability {
    return api::capability("pc.has_unsupported_probability_sources");
}

} // namespace capability

[[nodiscard]] inline auto make_descriptor(shape_summary const &shape = {}) -> api::ir_backend_descriptor {
    api::feature_set properties;
    if (shape.acyclic) {
        properties.add(capability::acyclic());
    }
    if (shape.has_csr_edge_view) {
        properties.add(capability::has_csr_edge_view());
    }
    if (shape.has_topological_view) {
        properties.add(capability::has_topological_view());
    }
    if (shape.has_kind_bucket_view) {
        properties.add(capability::has_kind_bucket_view());
    }
    for (probability_source_kind kind : shape.probability_source_kinds_present) {
        if (kind == probability_source_kind::literal) {
            properties.add(capability::has_literal_probability_sources());
        } else if (kind == probability_source_kind::parameter) {
            properties.add(capability::has_parameter_probability_sources());
        } else if (kind == probability_source_kind::unsupported) {
            properties.add(capability::has_unsupported_probability_sources());
        }
    }

    return api::ir_backend_descriptor{
        .provider = provider_id(),
        .backend = backend_id(),
        .storage_type = storage_type_id(),
        .kind = api::ir_kind::probabilistic_boolean_dag,
        .name = "FuzzyBits probabilistic circuit",
        .properties = std::move(properties),
    };
}

} // namespace fuzzybits::ir::pc
