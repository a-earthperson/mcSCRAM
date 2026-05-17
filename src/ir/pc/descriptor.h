#pragma once

#include "api/capabilities.h"
#include "api/ids.h"
#include "ir/descriptor.h"
#include "ir/forms.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"
#include "ir/pc/target.h"

#include <cstddef>
#include <vector>

namespace fuzzybits::ir::pc {

    struct shape_summary final {
        std::size_t node_count = 0;
        std::size_t edge_count = 0;
        bool acyclic = true;
        bool has_csr_edge_view = false;
        bool has_unsupported_probability_sources = false;
        std::vector<op_code> op_codes_present;
        std::vector<probability_source_kind> probability_source_kinds_present;
        std::vector<observable_role> observable_roles_present;
    };

    [[nodiscard]] inline auto storage_type_id() -> api::type_id {
        return api::type_id("ir.pc.circuit");
    }
    [[nodiscard]] inline auto endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("ir.pc");
    }

    [[nodiscard]] inline auto make_descriptor(shape_summary const & = {}) -> ir::descriptor {
        return ir::descriptor{
            .provider = api::provider_id("fuzzybits"),
            .endpoint = endpoint_id(),
            .storage_type = storage_type_id(),
            .form = forms::pc(),
            .name = "probabilistic circuit",
            .capabilities =
                api::capability_set{
                    api::capability_id("ir.probabilistic_boolean_dag"),
                    api::capability_id("ir.explicit_edge_polarity"),
                    api::capability_id("ir.targets"),
                    api::capability_id("ir.literal_probabilities"),
                },
        };
    }

} // namespace fuzzybits::ir::pc
