#include "ir/pc/edge.h"
#include "ir/pc/builder.h"
#include "ir/forms.h"

#include <utility>

namespace {

    [[nodiscard]] auto builds_valid_circuit() -> bool {
        fuzzybits::ir::pc::builder builder;
        const auto probability = builder.add_literal_probability(0.25);
        const auto event = builder.add_basic_event(probability);
        const auto constant = builder.add_constant(true);
        const auto gate = builder.add_and({
            fuzzybits::ir::pc::edge_ref{.source = event},
            fuzzybits::ir::pc::edge_ref{.source = constant},
        });
        const auto target = builder.add_target(gate);
        (void)target;

        const auto result = std::move(builder).seal();
        if (!result.ok()) {
            return false;
        }
        auto const &circuit = result.output;
        return circuit.valid() && circuit.node_count() == 3 && circuit.edge_count() == 2 &&
               circuit.descriptor().form == fuzzybits::ir::forms::pc() && circuit.to_handle().valid();
    }

    [[nodiscard]] auto rejects_bad_probability() -> bool {
        fuzzybits::ir::pc::builder builder;
        const auto probability = builder.add_literal_probability(2.0);
        const auto event = builder.add_basic_event(probability);
        const auto target = builder.add_target(event);
        (void)target;

        const auto result = std::move(builder).seal();
        return !result.ok() && result.diagnostics.has_errors();
    }

} // namespace

int main() {
    if (!builds_valid_circuit()) {
        return 1;
    }
    if (!rejects_bad_probability()) {
        return 2;
    }
    return 0;
}
