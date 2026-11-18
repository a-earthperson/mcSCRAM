#include "ir/pc/builder.h"
#include "io/ftp/detail/model.h"

#include "api/diagnostics.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {
    namespace {

        class PcBuilder final {
        public:
            explicit PcBuilder(ParsedModel const &model) : model_(model) {}

            [[nodiscard]] auto build(std::vector<std::string> const &targets) -> ir::pc::finalize_result {
                for (auto const &target : targets) {
                    if (const auto node = buildNode(target); node.valid()) {
                        const auto targetId = builder_.add_target(node);
                        (void)targetId;
                    }
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
            [[nodiscard]] auto buildNode(std::string const &name) -> ir::pc::bool_node_id {
                if (const auto built = builtNodes_.find(name); built != builtNodes_.end()) {
                    return built->second;
                }
                if (visiting_.contains(name)) {
                    diagnostics_.push(api::make_error("io.ftp.cycle", "FTP gate cycle involving '" + name + "'"));
                    return ir::pc::bool_node_id::invalid();
                }

                if (const auto definition = model_.definitionByName.find(name); definition != model_.definitionByName.end()) {
                    return buildDefinition(model_.definitions[definition->second]);
                }
                return buildBasicEvent(name);
            }

            [[nodiscard]] auto buildDefinition(Definition const &definition) -> ir::pc::bool_node_id {
                visiting_.insert(definition.name);

                std::vector<ir::pc::edge_ref> operands;
                operands.reserve(definition.operands.size());
                for (const auto &[name, complement] : definition.operands) {
                    const auto operand = buildNode(name);
                    if (!operand.valid()) {
                        visiting_.erase(definition.name);
                        return ir::pc::bool_node_id::invalid();
                    }
                    operands.push_back(ir::pc::edge_ref{
                        .source = operand,
                        .complement = complement,
                    });
                }

                ir::pc::bool_node_id node;
                switch (definition.kind) {
                case DefinitionKind::and_gate:
                    node = builder_.add_and(operands);
                    break;
                case DefinitionKind::or_gate:
                    node = builder_.add_or(operands);
                    break;
                case DefinitionKind::at_least:
                    node = builder_.add_cardinality(definition.threshold, static_cast<std::uint32_t>(operands.size()), operands);
                    break;
                }

                visiting_.erase(definition.name);
                builtNodes_.emplace(definition.name, node);
                return node;
            }

            [[nodiscard]] auto buildBasicEvent(std::string const &name) -> ir::pc::bool_node_id {
                ir::pc::probability_source_id probability;
                if (const auto found = model_.probabilities.find(name); found != model_.probabilities.end()) {
                    if (std::isfinite(found->second) && found->second >= 0.0 && found->second <= 1.0) {
                        probability = builder_.add_literal_probability(found->second);
                    } else {
                        probability = builder_.add_parameter_probability(name);
                        diagnostics_.push(api::make_warning("io.ftp.non-probability-import",
                                                            "imported value for basic event '" + name +
                                                                "' is outside the literal probability range [0, 1]"));
                    }
                } else {
                    probability = builder_.add_parameter_probability(name);
                    diagnostics_.push(api::make_warning("io.ftp.missing-probability",
                                                        "basic event '" + name + "' has no literal probability in the FTP file"));
                }

                const auto node = builder_.add_basic_event(probability);
                builtNodes_.emplace(name, node);
                return node;
            }

            ParsedModel const &model_;
            ir::pc::builder builder_;
            api::diagnostic_list diagnostics_;
            std::unordered_map<std::string, ir::pc::bool_node_id> builtNodes_;
            std::unordered_set<std::string> visiting_;
        };

    } // namespace

    auto build_pc(ParsedModel const &model, std::vector<std::string> const &targets) -> ir::pc::finalize_result {
        return PcBuilder(model).build(targets);
    }

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
