#include "io/aiger/descriptor.h"

#include "api/capabilities.h"
#include "api/diagnostics.h"
#include "api/ids.h"
#include "api/options.h"
#include "api/result.h"
#include "io/descriptor.h"
#include "io/formats.h"
#include "ir/forms.h"
#include "ir/handle.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <ostream>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::aiger {
    namespace {

        enum class VisitState : std::uint8_t {
            unseen,
            visiting,
            done,
        };

        struct Layout final {
            std::vector<ir::pc::bool_node_id> inputs;
            std::vector<ir::pc::bool_node_id> ands;
            // AIGER variable indices are reconstructed for serialization. They are
            // not PC node ids: inputs are assigned first, then ANDs, because binary
            // AIGER requires that compact variable layout.
            std::vector<std::uint32_t> inputVariableByNode;
            std::vector<std::uint32_t> andVariableByNode;
        };

        [[nodiscard]] auto isPrintableSymbol(std::string const &symbol) -> bool {
            return !symbol.empty() && std::ranges::all_of(symbol, [](const unsigned char character) {
                return character != '\n' && character != '\r' && std::isprint(character) != 0;
            });
        }

        [[nodiscard]] auto dependentCounts(ir::pc::circuit const &model) -> std::vector<std::uint32_t> {
            std::vector<std::uint32_t> counts(model.node_count(), 0);
            for (std::uint32_t node = 0; node < model.node_count(); ++node) {
                for (auto const &edge : model.operands(ir::pc::bool_node_id(node))) {
                    ++counts[edge.source.raw()];
                }
            }
            return counts;
        }

        void validateProbabilitySource(ir::pc::circuit const &model, std::uint32_t const node, api::diagnostic_list &diagnostics) {
            const auto source = model.probability_source_refs()[node];
            if (!source.valid() || source.raw() >= model.probability_sources().size()) {
                diagnostics.push(api::make_error("io.aiger.invalid-input", "AIGER writer requires every input to have a probability source"));
                return;
            }
            auto const &probability = model.probability_sources()[source.raw()];
            if (probability.kind != ir::pc::probability_source_kind::parameter) {
                // AIGER has Boolean inputs and an optional symbol table; it has no
                // representation for literal probabilities or unsupported
                // probability payloads. Emitting such inputs would lose PC data.
                diagnostics.push(api::make_error("io.aiger.lossy-probability",
                                                 "AIGER writer can serialize only parameter-named inputs without losing probability payloads"));
                return;
            }
            if (!isPrintableSymbol(probability.parameter_name)) {
                diagnostics.push(api::make_error("io.aiger.invalid-symbol", "AIGER input symbol must be non-empty printable ASCII"));
            }
        }

        [[nodiscard]] auto validateSerializable(ir::pc::circuit const &model) -> api::diagnostic_list {
            api::diagnostic_list diagnostics;
            if (!model.valid()) {
                diagnostics.push(api::make_error("io.aiger.invalid-ir", "AIGER writer requires a valid PC circuit"));
                return diagnostics;
            }
            if (!model.form_profile().is_aig()) {
                // The IO backend is not allowed to compile general Boolean DAGs into
                // AIGs. Even if a circuit could be rewritten using De Morgan or gate
                // decomposition, that transform belongs in compiler/IR code with a
                // transform report.
                diagnostics.push(api::make_error("io.aiger.non-aig-form", "AIGER writer requires a PC circuit that already satisfies ir.aig"));
                return diagnostics;
            }

            const auto dependents = dependentCounts(model);
            std::vector<std::uint8_t> targetNodes(model.node_count(), 0);
            for (auto const &target : model.targets()) {
                if (target.node.valid() && target.node.raw() < targetNodes.size()) {
                    targetNodes[target.node.raw()] = 1;
                }
            }
            for (std::uint32_t node = 0; node < model.node_count(); ++node) {
                const auto code = model.op_codes()[node];
                switch (code) {
                case ir::pc::op_code::basic_event:
                    validateProbabilitySource(model, node, diagnostics);
                    break;
                case ir::pc::op_code::constant:
                    break;
                case ir::pc::op_code::and_gate:
                    if (model.operand_counts()[node] != 2) {
                        diagnostics.push(api::make_error("io.aiger.unsupported-arity", "AIGER writer requires every AND gate to have fan-in 2"));
                    }
                    break;
                case ir::pc::op_code::not_gate:
                    if (model.operand_counts()[node] != 1 || dependents[node] != 0 || targetNodes[node] == 0) {
                        // Internal NOT nodes would have to be folded into every
                        // outgoing edge or represented by invented AND structure.
                        // Only target-only NOT nodes are accepted because AIGER
                        // outputs are polarized literals, so this is a direct
                        // serialization of already-present PC shape.
                        diagnostics.push(api::make_error(
                            "io.aiger.unsupported-not",
                            "AIGER writer can serialize NOT nodes only as target-edge inversions already present in an AIG-shaped PC"));
                    }
                    break;
                case ir::pc::op_code::house_event:
                case ir::pc::op_code::or_gate:
                case ir::pc::op_code::xor_gate:
                case ir::pc::op_code::nand_gate:
                case ir::pc::op_code::nor_gate:
                case ir::pc::op_code::xnor_gate:
                case ir::pc::op_code::cardinality:
                    diagnostics.push(api::make_error("io.aiger.unsupported-opcode",
                                                     "AIGER writer cannot serialize this PC opcode without compiling the circuit"));
                    break;
                }
            }
            return diagnostics;
        }

        void visitAnd(ir::pc::circuit const &model, ir::pc::bool_node_id node, std::vector<VisitState> &states, Layout &layout,
                      api::diagnostic_list &diagnostics) {
            const auto raw = node.raw();
            if (states[raw] == VisitState::done) {
                return;
            }
            if (states[raw] == VisitState::visiting) {
                diagnostics.push(api::make_error("io.aiger.cycle", "PC circuit contains a cycle while building AIGER layout"));
                return;
            }
            states[raw] = VisitState::visiting;
            for (auto const &edge : model.operands(node)) {
                if (model.op_codes()[edge.source.raw()] == ir::pc::op_code::and_gate) {
                    visitAnd(model, edge.source, states, layout, diagnostics);
                }
            }
            states[raw] = VisitState::done;
            // The emitted AND order is dependency-first, not necessarily PC node-id
            // order or original file order. This preserves semantics and satisfies
            // binary AIGER ordering, but it can change delta sizes and byte-level
            // output compared with the source artifact.
            layout.ands.push_back(node);
        }

        [[nodiscard]] auto makeLayout(ir::pc::circuit const &model) -> api::result<Layout> {
            auto diagnostics = validateSerializable(model);
            if (diagnostics.has_errors()) {
                return api::result<Layout>::fail(std::move(diagnostics));
            }

            Layout layout;
            layout.inputVariableByNode.assign(model.node_count(), 0);
            layout.andVariableByNode.assign(model.node_count(), 0);
            for (std::uint32_t node = 0; node < model.node_count(); ++node) {
                if (model.op_codes()[node] == ir::pc::op_code::basic_event) {
                    // Input order follows existing PC node order. We do not try to
                    // recover prior AIGER variable numbers from symbols or names, so
                    // round-trips are structurally stable for this writer but not
                    // byte-identical to arbitrary source ordering.
                    layout.inputs.push_back(ir::pc::bool_node_id(node));
                    layout.inputVariableByNode[node] = static_cast<std::uint32_t>(layout.inputs.size());
                }
            }

            std::vector<VisitState> states(model.node_count(), VisitState::unseen);
            for (std::uint32_t node = 0; node < model.node_count(); ++node) {
                if (model.op_codes()[node] == ir::pc::op_code::and_gate) {
                    visitAnd(model, ir::pc::bool_node_id(node), states, layout, diagnostics);
                }
            }
            if (diagnostics.has_errors()) {
                return api::result<Layout>::fail(std::move(diagnostics));
            }
            if (static_cast<std::uint64_t>(layout.inputs.size()) + layout.ands.size() >
                ((std::numeric_limits<std::uint32_t>::max() - 1U) >> 1U)) {
                return api::result<Layout>::fail(
                    api::make_error("io.aiger.literal-overflow", "AIGER writer output would exceed the 32-bit literal range"));
            }

            const auto inputCount = static_cast<std::uint32_t>(layout.inputs.size());
            for (std::uint32_t index = 0; index < layout.ands.size(); ++index) {
                layout.andVariableByNode[layout.ands[index].raw()] = inputCount + index + 1U;
            }
            return api::result<Layout>::ok(std::move(layout));
        }

        [[nodiscard]] auto literalForNode(ir::pc::circuit const &model, Layout const &layout, ir::pc::bool_node_id node) -> std::uint32_t {
            switch (model.op_codes()[node.raw()]) {
            case ir::pc::op_code::constant:
                return model.constant_values()[node.raw()] != 0 ? 1U : 0U;
            case ir::pc::op_code::basic_event:
                return 2U * layout.inputVariableByNode[node.raw()];
            case ir::pc::op_code::and_gate:
                return 2U * layout.andVariableByNode[node.raw()];
            case ir::pc::op_code::not_gate:
            case ir::pc::op_code::house_event:
            case ir::pc::op_code::or_gate:
            case ir::pc::op_code::xor_gate:
            case ir::pc::op_code::nand_gate:
            case ir::pc::op_code::nor_gate:
            case ir::pc::op_code::xnor_gate:
            case ir::pc::op_code::cardinality:
                return 0;
            }
            return 0;
        }

        [[nodiscard]] auto literalForEdge(ir::pc::circuit const &model, Layout const &layout, ir::pc::edge_ref const edge) -> std::uint32_t {
            return literalForNode(model, layout, edge.source) ^ (edge.complement ? 1U : 0U);
        }

        [[nodiscard]] auto literalForTarget(ir::pc::circuit const &model, Layout const &layout, ir::pc::bool_node_id node) -> std::uint32_t {
            if (model.op_codes()[node.raw()] != ir::pc::op_code::not_gate) {
                return literalForNode(model, layout, node);
            }
            // This reverses the reader's explicit target NOT materialization back
            // into AIGER's native polarized output literal without changing any
            // internal graph structure.
            const auto operand = model.operands(node).front();
            return literalForEdge(model, layout, operand) ^ 1U;
        }

        void encodeDelta(std::ostream &output, std::uint32_t value) {
            while ((value & ~0x7FU) != 0) {
                output.put(static_cast<char>((value & 0x7FU) | 0x80U));
                value >>= 7U;
            }
            output.put(static_cast<char>(value));
        }

        void writeInputSymbols(ir::pc::circuit const &model, Layout const &layout, std::ostream &output) {
            for (std::uint32_t index = 0; index < layout.inputs.size(); ++index) {
                const auto node = layout.inputs[index].raw();
                const auto source = model.probability_source_refs()[node];
                // Parameter names are emitted as input symbols. This is the only
                // probability-source payload AIGER can preserve in the current
                // mapping, and it is why literal probabilities are rejected above.
                output << 'i' << index << ' ' << model.probability_sources()[source.raw()].parameter_name << '\n';
            }
        }

        [[nodiscard]] auto writePc(ir::pc::circuit const &model, std::ostream &output, io::write_request const &request)
            -> io::write_result {
            if (!request.format.empty() && request.format != formats::aiger()) {
                return io::write_result::fail(api::make_error("io.aiger.format-mismatch", "AIGER writer received a non-AIGER format request"));
            }

            auto layoutResult = makeLayout(model);
            if (!layoutResult) {
                return io::write_result::fail(layoutResult.diagnostics());
            }
            auto layout = std::move(layoutResult).value();

            const auto inputCount = static_cast<std::uint32_t>(layout.inputs.size());
            const auto andCount = static_cast<std::uint32_t>(layout.ands.size());
            const auto maxVariable = inputCount + andCount;
            // Always emit binary AIGER. The writer chooses a compact legal layout from
            // the PC shape rather than preserving ASCII-vs-binary source syntax, unused
            // variable gaps, comments, or symbol ordering from an earlier file.
            output << "aig " << maxVariable << ' ' << inputCount << " 0 " << model.targets().size() << ' ' << andCount << '\n';
            for (auto const &target : model.targets()) {
                output << literalForTarget(model, layout, target.node) << '\n';
            }
            for (auto const node : layout.ands) {
                const auto lhs = literalForNode(model, layout, node);
                auto rhs0 = literalForEdge(model, layout, model.operands(node)[0]);
                auto rhs1 = literalForEdge(model, layout, model.operands(node)[1]);
                if (rhs0 < rhs1) {
                    std::swap(rhs0, rhs1);
                }
                encodeDelta(output, lhs - rhs0);
                encodeDelta(output, rhs0 - rhs1);
            }
            writeInputSymbols(model, layout, output);
            return io::write_result::ok(true);
        }

    } // namespace

    auto writer_endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("io.fuzzybits.aiger.writer");
    }

    auto writer_descriptor() -> io::writer_descriptor {
        return io::writer_descriptor{
            .name = "FuzzyBits AIGER writer",
            .provider = api::provider_id("fuzzybits"),
            .endpoint = writer_endpoint_id(),
            .output_format = formats::aiger(),
            .accepts =
                api::capability_set{
                    api::capability_id("ir.probabilistic_boolean_dag"),
                    api::capability_id("ir.explicit_edge_polarity"),
                    api::capability_id("ir.targets"),
                    api::capability_id("ir.parameter_probabilities"),
                },
            .options = api::option_schema{},
        };
    }

    auto write(ir::handle const &model, std::ostream &output) -> io::write_result {
        return write(model, output, io::write_request{.format = formats::aiger()});
    }

    auto write(ir::handle const &model, std::ostream &output, io::write_request const &request) -> io::write_result {
        if (!request.format.empty() && request.format != formats::aiger()) {
            return io::write_result::fail(api::make_error("io.aiger.format-mismatch", "AIGER writer received a non-AIGER format request"));
        }

        auto const *pc = ir::pc::circuit_from(model);
        if (pc == nullptr) {
            return io::write_result::fail(api::make_error("io.aiger.unsupported-ir", "AIGER writer requires a PC IR handle"));
        }
        return writePc(*pc, output, request);
    }

} // namespace fuzzybits::io::backend::fuzzybits::aiger
