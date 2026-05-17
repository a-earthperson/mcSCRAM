#include "ir/pc/edge.h"
#include "ir/pc/circuit.h"
#include "ir/pc/builder.h"

#include <utility>

int main() {
    fuzzybits::ir::pc::builder circuitBuilder;
    auto probability = circuitBuilder.add_literal_probability(0.1);
    auto event = circuitBuilder.add_basic_event(probability);
    auto otherEvent = circuitBuilder.add_basic_event(probability);
    auto gate = circuitBuilder.add_and({
        fuzzybits::ir::pc::edge_ref{.source = event},
        fuzzybits::ir::pc::edge_ref{.source = otherEvent},
    });
    auto target = circuitBuilder.add_target(gate);
    (void)target;

    auto result = std::move(circuitBuilder).seal();
    if (!result.ok()) {
        return 1;
    }

    auto handle = result.output.to_handle();
    return fuzzybits::ir::pc::circuit_from(handle) == nullptr ? 1 : 0;
}
