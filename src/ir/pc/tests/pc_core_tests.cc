#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "ir/pc/edge.h"
#include "ir/pc/builder.h"
#include "ir/form_profile.h"
#include "ir/forms.h"

#include <algorithm>
#include <string>
#include <utility>

namespace {

    void buildsValidCircuit() {
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
        REQUIRE(result.ok());
        auto const &circuit = result.output;
        REQUIRE(circuit.valid());
        REQUIRE(circuit.node_count() == 3);
        REQUIRE(circuit.edge_count() == 2);
        REQUIRE(circuit.descriptor().form == fuzzybits::ir::forms::probabilistic_boolean_dag());
        REQUIRE(circuit.form_profile().is_bool_dag());
        REQUIRE(circuit.form_profile().is_probabilistic_boolean_dag());
        REQUIRE(circuit.satisfies_form(fuzzybits::ir::forms::bool_dag()));
        REQUIRE(circuit.to_handle().valid());
    }

    void rejectsBadProbability() {
        fuzzybits::ir::pc::builder builder;
        const auto probability = builder.add_literal_probability(2.0);
        const auto event = builder.add_basic_event(probability);
        const auto target = builder.add_target(event);
        (void)target;

        const auto result = std::move(builder).seal();
        REQUIRE_FALSE(result.ok());
        REQUIRE(result.diagnostics.has_errors());
    }

    void recordsFormHierarchy() {
        fuzzybits::ir::pc::builder builder;
        builder.set_form(fuzzybits::ir::forms::robdd());
        const auto probability = builder.add_literal_probability(0.25);
        const auto event = builder.add_basic_event(probability);
        const auto target = builder.add_target(event);
        (void)target;

        const auto result = std::move(builder).seal();
        REQUIRE(result.ok());

        auto const &circuit = result.output;
        const auto form_strings = circuit.satisfied_form_strings();
        const auto has_form_string = [&form_strings](std::string const &form) {
            return std::ranges::find(form_strings, form) != form_strings.end();
        };

        REQUIRE(circuit.form_profile().is_robdd());
        REQUIRE(circuit.form_profile().is_bdd());
        REQUIRE(circuit.form_profile().is_bool_dag());
        REQUIRE_FALSE(circuit.form_profile().is_cnf());
        REQUIRE(circuit.satisfies_form(fuzzybits::ir::forms::bdd()));
        REQUIRE(fuzzybits::ir::descriptor_satisfies_form(circuit.descriptor(), fuzzybits::ir::forms::bdd()));
        REQUIRE(fuzzybits::ir::descriptor_satisfies_form(circuit.descriptor(), fuzzybits::ir::forms::probabilistic_boolean_dag()));
        REQUIRE(has_form_string("ir.bool_dag"));
        REQUIRE(has_form_string("ir.bdd"));
        REQUIRE(has_form_string("ir.robdd"));
    }

} // namespace

TEST_CASE("PC builder seals a valid probabilistic boolean DAG", "[ir][pc][builder]") {
    buildsValidCircuit();
}

TEST_CASE("PC builder rejects out-of-domain literal probability", "[ir][pc][builder]") {
    rejectsBadProbability();
}

TEST_CASE("PC builder records satisfied form hierarchy on the circuit", "[ir][pc][builder]") {
    recordsFormHierarchy();
}
