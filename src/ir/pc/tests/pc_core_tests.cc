#include "ir/pc/edge.h"
#include "ir/pc/builder.h"
#include "ir/form_profile.h"
#include "ir/forms.h"

#include <algorithm>
#include <string>
#include <utility>

namespace {

    [[nodiscard]] auto buildsValidCircuit() -> bool {
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
               circuit.descriptor().form == fuzzybits::ir::forms::probabilistic_boolean_dag() &&
               circuit.form_profile().is_bool_dag() && circuit.form_profile().is_probabilistic_boolean_dag() &&
               circuit.satisfies_form(fuzzybits::ir::forms::bool_dag()) && circuit.to_handle().valid();
    }

    [[nodiscard]] auto rejectsBadProbability() -> bool {
        fuzzybits::ir::pc::builder builder;
        const auto probability = builder.add_literal_probability(2.0);
        const auto event = builder.add_basic_event(probability);
        const auto target = builder.add_target(event);
        (void)target;

        const auto result = std::move(builder).seal();
        return !result.ok() && result.diagnostics.has_errors();
    }

    [[nodiscard]] auto recordsFormHierarchy() -> bool {
        fuzzybits::ir::pc::builder builder;
        builder.set_form(fuzzybits::ir::forms::robdd());
        const auto probability = builder.add_literal_probability(0.25);
        const auto event = builder.add_basic_event(probability);
        const auto target = builder.add_target(event);
        (void)target;

        const auto result = std::move(builder).seal();
        if (!result.ok()) {
            return false;
        }

        auto const &circuit = result.output;
        const auto form_strings = circuit.satisfied_form_strings();
        const auto has_form_string = [&form_strings](std::string const &form) {
            return std::ranges::find(form_strings, form) != form_strings.end();
        };

        return circuit.form_profile().is_robdd() && circuit.form_profile().is_bdd() &&
               circuit.form_profile().is_bool_dag() && !circuit.form_profile().is_cnf() &&
               circuit.satisfies_form(fuzzybits::ir::forms::bdd()) &&
               fuzzybits::ir::descriptor_satisfies_form(circuit.descriptor(), fuzzybits::ir::forms::bdd()) &&
               fuzzybits::ir::descriptor_satisfies_form(circuit.descriptor(), fuzzybits::ir::forms::probabilistic_boolean_dag()) &&
               has_form_string("ir.bool_dag") && has_form_string("ir.bdd") && has_form_string("ir.robdd");
    }

} // namespace

int main() {
    if (!buildsValidCircuit()) {
        return 1;
    }
    if (!rejectsBadProbability()) {
        return 2;
    }
    if (!recordsFormHierarchy()) {
        return 3;
    }
    return 0;
}
