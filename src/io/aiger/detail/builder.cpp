#include "io/aiger/detail/model.h"

#include "ir/forms.h"
#include "ir/pc/builder.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"

#include "api/diagnostics.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace fuzzybits::io::backend::fuzzybits::aiger::detail {
    namespace {

        class PcBuilder final {
        public:
            explicit PcBuilder(ParsedModel const &model) : model_(model) {
                // Successful AIGER ingestion is already structurally AIG-shaped, so
                // the sealed PC is elevated to ir.aig here. No normalization or
                // Boolean rewriting is performed to earn that form.
                builder_.set_form(ir::forms::aig());
                definitionsByVariable_.reserve(model_.andDefinitions.size());
                for (auto const &definition : model_.andDefinitions) {
                    definitionsByVariable_.emplace(literalVariable(definition.lhs), &definition);
                }
            }

            [[nodiscard]] auto build() -> ir::pc::finalize_result {
                if (model_.diagnostics.has_errors()) {
                    return ir::pc::finalize_result{
                        .output = {},
                        .diagnostics = model_.diagnostics,
                    };
                }

                builder_.reserve_nodes(estimatedNodeCount());
                buildInputs();
                // Build every AIGER AND definition, including definitions that are
                // not reachable from an output. ASCII AIGER permits unused ANDs;
                // keeping them makes import/export a structural storage mapping,
                // not a reachability-reducing compiler pass.
                for (const auto &[lhs, rhs] : model_.andDefinitions) {
                    const auto node = buildVariable(literalVariable(lhs));
                    (void)node;
                }
                for (auto const literal : model_.outputLiterals) {
                    const auto target = builder_.add_target(buildLiteralNode(literal));
                    (void)target;
                }

                if (model_.inputLiterals.empty() && model_.andDefinitions.empty() && model_.outputLiterals.empty()) {
                    diagnostics_.push(api::make_error("io.aiger.empty-circuit", "empty AIGER files cannot be represented by the current PC storage"));
                }
                if (diagnostics_.has_errors()) {
                    return ir::pc::finalize_result{
                        .output = {},
                        .diagnostics = std::move(diagnostics_),
                    };
                }

                auto finalized = std::move(builder_).seal();
                finalized.diagnostics.append(std::move(diagnostics_));
                return finalized;
            }

        private:
            [[nodiscard]] auto estimatedNodeCount() const -> std::uint32_t {
                const auto count = static_cast<std::uint64_t>(model_.inputLiterals.size()) + model_.andDefinitions.size() +
                                   model_.outputLiterals.size() + 2U;
                if (count > std::numeric_limits<std::uint32_t>::max()) {
                    return std::numeric_limits<std::uint32_t>::max();
                }
                return static_cast<std::uint32_t>(count);
            }

            void buildInputs() {
                for (std::uint32_t index = 0; index < model_.inputLiterals.size(); ++index) {
                    const auto variable = literalVariable(model_.inputLiterals[index]);
                    // AIGER inputs are Boolean variables, not probability-bearing
                    // events. PC currently needs a probability source for each
                    // basic_event, so input symbols become parameter names and
                    // unnamed inputs receive deterministic synthetic names.
                    auto name = model_.inputSymbols[index].empty() ? "i" + std::to_string(index) : model_.inputSymbols[index];
                    const auto probability = builder_.add_parameter_probability(name);
                    const auto node = builder_.add_basic_event(probability);
                    nodeByVariable_.emplace(variable, node);
                }
            }

            [[nodiscard]] auto buildConstant(const bool value) -> ir::pc::bool_node_id {
                auto &cached = value ? trueConstant_ : falseConstant_;
                if (!cached.valid()) {
                    cached = builder_.add_constant(value);
                }
                return cached;
            }

            [[nodiscard]] auto buildLiteralNode(const std::uint32_t literal) -> ir::pc::bool_node_id {
                if (literal == 0) {
                    return buildConstant(false);
                }
                if (literal == 1) {
                    return buildConstant(true);
                }
                const auto source = buildVariable(literalVariable(literal));
                if (!source.valid()) {
                    return ir::pc::bool_node_id::invalid();
                }
                if ((literal & 1U) == 0) {
                    return source;
                }
                // PC targets reference nodes rather than polarized edges. A
                // complemented AIGER output therefore becomes an explicit NOT node,
                // increasing PC node count by at most one per complemented output.
                return builder_.add_not(ir::pc::edge_ref{.source = source});
            }

            [[nodiscard]] auto buildLiteralEdge(const std::uint32_t literal) -> ir::pc::edge_ref {
                if (literal == 0) {
                    return ir::pc::edge_ref{.source = buildConstant(false)};
                }
                if (literal == 1) {
                    return ir::pc::edge_ref{.source = buildConstant(true)};
                }
                return ir::pc::edge_ref{
                    .source = buildVariable(literalVariable(literal)),
                    // AND fan-in polarity maps directly to PC edge polarity. This
                    // is the lossless path; we do not synthesize separate NOT nodes
                    // for internal AIGER literals.
                    .complement = (literal & 1U) != 0,
                };
            }

            [[nodiscard]] auto buildVariable(const std::uint32_t variable) -> ir::pc::bool_node_id {
                if (const auto found = nodeByVariable_.find(variable); found != nodeByVariable_.end()) {
                    return found->second;
                }
                if (visitingVariables_.contains(variable)) {
                    diagnostics_.push(api::make_error("io.aiger.cycle", "AIGER AND dependency cycle involving variable " + std::to_string(variable)));
                    return ir::pc::bool_node_id::invalid();
                }
                const auto definition = definitionByVariable(variable);
                if (definition == nullptr) {
                    diagnostics_.push(api::make_error("io.aiger.undefined-literal",
                                                       "AIGER literal references undefined variable " + std::to_string(variable)));
                    return ir::pc::bool_node_id::invalid();
                }

                visitingVariables_.insert(variable);
                const std::array operands{
                    buildLiteralEdge(definition->rhs[0]),
                    buildLiteralEdge(definition->rhs[1]),
                };
                visitingVariables_.erase(variable);

                if (!operands[0].source.valid() || !operands[1].source.valid()) {
                    return ir::pc::bool_node_id::invalid();
                }
                const auto node = builder_.add_and(operands);
                nodeByVariable_.emplace(variable, node);
                return node;
            }

            [[nodiscard]] auto definitionByVariable(const std::uint32_t variable) const -> AndDefinition const * {
                const auto found = definitionsByVariable_.find(variable);
                return found == definitionsByVariable_.end() ? nullptr : found->second;
            }

            ParsedModel const &model_;
            ir::pc::builder builder_;
            api::diagnostic_list diagnostics_;
            std::unordered_map<std::uint32_t, ir::pc::bool_node_id> nodeByVariable_;
            std::unordered_map<std::uint32_t, AndDefinition const *> definitionsByVariable_;
            std::unordered_set<std::uint32_t> visitingVariables_;
            ir::pc::bool_node_id falseConstant_;
            ir::pc::bool_node_id trueConstant_;
        };

    } // namespace

    auto build_pc(ParsedModel const &model) -> ir::pc::finalize_result {
        return PcBuilder(model).build();
    }

} // namespace fuzzybits::io::backend::fuzzybits::aiger::detail
