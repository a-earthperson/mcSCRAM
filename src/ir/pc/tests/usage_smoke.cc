#include "ir/pc/pc.h"

#include <utility>

int main() {
    fuzzybits::ir::pc::builder circuit_builder;
    auto probability = circuit_builder.add_literal_probability(0.1);
    auto event = circuit_builder.add_basic_event(probability);
    auto other_event = circuit_builder.add_basic_event(probability);
    auto gate = circuit_builder.add_and({
        fuzzybits::ir::pc::edge_ref{.source = event},
        fuzzybits::ir::pc::edge_ref{.source = other_event},
    });
    auto target = circuit_builder.add_target(gate);
    (void)target;

    auto result = std::move(circuit_builder).seal();
    if (!result.ok()) {
        return 1;
    }

    auto handle = result.output.to_ir_handle();
    return fuzzybits::ir::pc::circuit_from(handle) == nullptr ? 1 : 0;
}
