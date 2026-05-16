#include "ir/pc/pc.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

class test_failure final : public std::exception {
  public:
    explicit test_failure(std::string message) : message_(std::move(message)) {}

    [[nodiscard]] auto what() const noexcept -> char const * override { return message_.c_str(); }

  private:
    std::string message_;
};

void require(bool condition, std::string message) {
    if (!condition) {
        throw test_failure(std::move(message));
    }
}

template <typename value_type>
[[nodiscard]] auto contains(std::span<value_type const> values, value_type expected) -> bool {
    for (value_type value : values) {
        if (value == expected) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] auto diagnostic_messages(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
    std::string joined;
    for (fuzzybits::api::diagnostic const &diagnostic : diagnostics) {
        if (!joined.empty()) {
            joined += '\n';
        }
        joined += diagnostic.code.str();
        joined += ": ";
        joined += diagnostic.message;
    }
    return joined;
}

[[nodiscard]] auto has_diagnostic(fuzzybits::api::diagnostic_list const &diagnostics, std::string_view code) -> bool {
    for (fuzzybits::api::diagnostic const &diagnostic : diagnostics) {
        if (diagnostic.code.view() == code) {
            return true;
        }
    }
    return false;
}

void builds_literal_probability_circuit_and_ir_handle() {
    namespace pc = fuzzybits::ir::pc;

    pc::builder builder;
    builder.reserve_nodes(4);
    const auto probability = builder.add_literal_probability(0.25);
    const auto event = builder.add_basic_event(probability);
    const auto constant = builder.add_constant(true);

    const std::vector<pc::edge_ref> operands{
        pc::edge_ref{.source = event},
        pc::edge_ref{.source = constant, .complement = true},
    };
    const auto gate = builder.add_gate(pc::op_code::and_gate, operands);
    const auto target = builder.add_target(gate, pc::observable_role::primary);

    auto result = std::move(builder).seal();
    require(result.ok(), "literal circuit should seal without diagnostics:\n" + diagnostic_messages(result.diagnostics));

    pc::circuit const &circuit = result.output;
    require(circuit.valid(), "sealed circuit should be valid");
    require(circuit.node_count() == 3, "sealed circuit should contain three nodes");
    require(circuit.edge_count() == 2, "sealed circuit should contain two edges");
    require(circuit.op_codes()[event.raw()] == pc::op_code::basic_event, "event opcode should be basic_event");
    require(circuit.op_codes()[constant.raw()] == pc::op_code::constant, "constant opcode should be constant");
    require(circuit.op_codes()[gate.raw()] == pc::op_code::and_gate, "gate opcode should be and_gate");
    require(circuit.probability_source_refs()[event.raw()] == probability,
            "basic event should reference the literal probability source");
    require(circuit.constant_values()[constant.raw()] == 1U, "constant true should be recorded as 1");
    require(circuit.targets().size() == 1, "circuit should expose one target");
    require(circuit.targets().front().id == target, "target id should be stable in target table");
    require(circuit.targets().front().node == gate, "target should reference the gate");

    auto gate_operands = circuit.operands(gate);
    require(gate_operands.size() == 2, "gate operand view should have two edges");
    require(gate_operands[0].source == event && !gate_operands[0].complement,
            "first gate operand should reference event positively");
    require(gate_operands[1].source == constant && gate_operands[1].complement,
            "second gate operand should reference constant negatively");

    require(circuit.shape().acyclic, "shape should record acyclicity after successful finalization");
    require(circuit.shape().has_csr_edge_view, "R1 circuit should expose CSR edge view shape");
    require(contains<pc::op_code>(circuit.shape().op_codes_present, pc::op_code::and_gate),
            "shape should record present opcodes");
    require(contains<pc::probability_source_kind>(circuit.shape().probability_source_kinds_present,
                                                  pc::probability_source_kind::literal),
            "shape should record literal probability sources");

    auto handle = circuit.to_ir_handle();
    require(handle.valid(), "valid circuit should wrap as api::ir_handle");
    require(handle.storage_type() == pc::storage_type_id(), "ir_handle should carry PC storage type");
    require(handle.descriptor().kind == fuzzybits::api::ir_kind::probabilistic_boolean_dag,
            "PC descriptor should declare probabilistic Boolean DAG kind");
    require(handle.descriptor().properties.contains(pc::capability::acyclic()),
            "PC descriptor should expose acyclic capability");
    require(pc::is_pc(handle), "PC handle predicate should recognize wrapped circuit");
    require(pc::circuit_from(handle) != nullptr, "PC handle extractor should return wrapped circuit");
}

void accepts_acyclic_forward_references_without_requiring_topological_ids() {
    namespace pc = fuzzybits::ir::pc;

    pc::builder builder;
    const std::vector<pc::edge_ref> operands{
        pc::edge_ref{.source = pc::bool_node_id(1)},
    };
    const auto gate = builder.add_gate(pc::op_code::not_gate, operands);
    const auto constant = builder.add_constant(false);
    const auto target = builder.add_target(gate, pc::observable_role::watch);
    (void)target;

    auto result = std::move(builder).seal();
    require(result.ok(),
            "forward-referenced acyclic circuit should seal:\n" + diagnostic_messages(result.diagnostics));
    require(result.output.node_count() == 2, "forward-reference circuit should contain two nodes");
    require(result.output.operands(gate).front().source == constant,
            "forward reference should resolve to the later node id");
    require(!result.output.descriptor().properties.contains(pc::capability::has_topological_view()),
            "R1 base circuit should not claim topological view capability by construction");
}

void supports_cardinality_and_parameter_probability_sources() {
    namespace pc = fuzzybits::ir::pc;

    pc::builder builder;
    const auto parameter = builder.add_parameter_probability("pump_failure_probability");
    const auto a = builder.add_basic_event(parameter);
    const auto b = builder.add_constant(true);
    const auto c = builder.add_constant(false);
    const std::vector<pc::edge_ref> operands{
        pc::edge_ref{.source = a},
        pc::edge_ref{.source = b},
        pc::edge_ref{.source = c, .complement = true},
    };
    const auto vote = builder.add_cardinality(2, 3, operands);
    const auto target = builder.add_target(vote, pc::observable_role::projection_anchor);
    (void)target;

    auto result = std::move(builder).seal();
    require(result.ok(), "cardinality circuit should seal:\n" + diagnostic_messages(result.diagnostics));
    require(result.output.op_codes()[vote.raw()] == pc::op_code::cardinality,
            "cardinality node should use the semantic cardinality opcode");
    require(result.output.cardinality_low()[vote.raw()] == 2, "cardinality low bound should be retained");
    require(result.output.cardinality_high()[vote.raw()] == 3, "cardinality high bound should be retained");
    require(result.output.probability_sources()[parameter.raw()].parameter_name == "pump_failure_probability",
            "parameter probability source name should be retained");
    require(result.output.descriptor().properties.contains(pc::capability::has_parameter_probability_sources()),
            "descriptor should expose parameter probability capability");
}

void preserves_unsupported_probability_sources_as_planning_visible_shape() {
    namespace pc = fuzzybits::ir::pc;

    pc::builder builder;
    const auto unsupported = builder.add_unsupported_probability("MEF random deviate requires expression RFC");
    const auto event = builder.add_basic_event(unsupported);
    const auto target = builder.add_target(event);
    (void)target;

    auto result = std::move(builder).seal();
    require(result.ok(), "unsupported probability placeholders should not block sealing");
    require(result.output.shape().has_unsupported_probability_sources,
            "shape should mark unsupported probability placeholders");
    require(result.output.descriptor().properties.contains(pc::capability::has_unsupported_probability_sources()),
            "descriptor should expose unsupported probability capability");
    require(result.output.probability_sources()[unsupported.raw()].diagnostic ==
                "MEF random deviate requires expression RFC",
            "unsupported probability diagnostic should be retained");
}

void reports_multiple_semantic_validation_failures() {
    namespace pc = fuzzybits::ir::pc;

    pc::builder builder;
    const auto invalid_literal = builder.add_literal_probability(1.25);
    const auto empty_parameter = builder.add_parameter_probability("");
    const auto bad_event = builder.add_basic_event(pc::probability_source_id::invalid());
    const auto literal_event = builder.add_basic_event(invalid_literal);
    const auto parameter_event = builder.add_basic_event(empty_parameter);
    const std::vector<pc::edge_ref> no_operands;
    const auto invalid_not = builder.add_gate(pc::op_code::not_gate, no_operands);
    const auto invalid_and = builder.add_gate(pc::op_code::and_gate, no_operands);
    const std::vector<pc::edge_ref> cardinality_operands{
        pc::edge_ref{.source = literal_event},
    };
    const auto invalid_cardinality = builder.add_cardinality(2, 1, cardinality_operands);
    const std::vector<pc::edge_ref> invalid_operands{
        pc::edge_ref{.source = pc::bool_node_id(99)},
    };
    const auto invalid_edge_gate = builder.add_gate(pc::op_code::or_gate, invalid_operands);
    const auto invalid_target = builder.add_target(pc::bool_node_id(123));
    const auto parameter_target = builder.add_target(parameter_event, pc::observable_role::watch);
    const auto bad_event_target = builder.add_target(bad_event, pc::observable_role::projection_anchor);
    (void)invalid_not;
    (void)invalid_and;
    (void)invalid_cardinality;
    (void)invalid_edge_gate;
    (void)invalid_target;
    (void)parameter_target;
    (void)bad_event_target;

    auto result = std::move(builder).seal();
    require(!result.ok(), "invalid circuit should fail finalization");
    require(!result.output.valid(), "failed finalization should not produce a valid circuit");
    require(has_diagnostic(result.diagnostics, "pc.invalid-literal-probability"),
            "invalid literal probability should be diagnosed");
    require(has_diagnostic(result.diagnostics, "pc.invalid-parameter-probability"),
            "empty parameter probability should be diagnosed");
    require(has_diagnostic(result.diagnostics, "pc.invalid-probability-source"),
            "basic event with invalid source should be diagnosed");
    require(has_diagnostic(result.diagnostics, "pc.invalid-arity"), "invalid gate arity should be diagnosed");
    require(has_diagnostic(result.diagnostics, "pc.invalid-cardinality"),
            "invalid cardinality bounds should be diagnosed");
    require(has_diagnostic(result.diagnostics, "pc.invalid-edge"), "invalid edge endpoint should be diagnosed");
    require(has_diagnostic(result.diagnostics, "pc.invalid-target"), "invalid target endpoint should be diagnosed");
}

void detects_cycles_even_when_all_edge_endpoints_exist() {
    namespace pc = fuzzybits::ir::pc;

    pc::builder builder;
    const std::vector<pc::edge_ref> first_operands{
        pc::edge_ref{.source = pc::bool_node_id(1)},
    };
    const auto first = builder.add_gate(pc::op_code::and_gate, first_operands);
    const std::vector<pc::edge_ref> second_operands{
        pc::edge_ref{.source = pc::bool_node_id(0)},
    };
    const auto second = builder.add_gate(pc::op_code::or_gate, second_operands);
    (void)first;
    (void)second;

    auto result = std::move(builder).seal();
    require(!result.ok(), "cyclic circuit should fail finalization");
    require(has_diagnostic(result.diagnostics, "pc.cycle"), "cycle should be diagnosed");
}

void rejects_empty_circuits() {
    namespace pc = fuzzybits::ir::pc;

    pc::builder builder;
    auto result = std::move(builder).seal();
    require(!result.ok(), "empty circuit should fail finalization");
    require(has_diagnostic(result.diagnostics, "pc.empty-circuit"), "empty circuit should be diagnosed");
}

using test_fn = void (*)();

struct test_case final {
    std::string_view name;
    test_fn run;
};

} // namespace

int main() {
    const test_case tests[] = {
        {"builds literal probability circuit and ir handle", builds_literal_probability_circuit_and_ir_handle},
        {"accepts acyclic forward references without topological ids",
         accepts_acyclic_forward_references_without_requiring_topological_ids},
        {"supports cardinality and parameter probability sources", supports_cardinality_and_parameter_probability_sources},
        {"preserves unsupported probability sources as planning-visible shape",
         preserves_unsupported_probability_sources_as_planning_visible_shape},
        {"reports multiple semantic validation failures", reports_multiple_semantic_validation_failures},
        {"detects cycles even when all edge endpoints exist", detects_cycles_even_when_all_edge_endpoints_exist},
        {"rejects empty circuits", rejects_empty_circuits},
    };

    int failures = 0;
    for (test_case const &test : tests) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (std::exception const &error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    }

    return failures == 0 ? 0 : 1;
}
