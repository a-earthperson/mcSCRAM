#include "io/ftp/descriptor.h"

#include "api/capabilities.h"
#include "api/diagnostics.h"
#include "api/ids.h"
#include "api/options.h"
#include "api/result.h"
#include "io/descriptor.h"
#include "io/formats.h"
#include "ir/pc/builder.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp {
    namespace {

        enum class DefinitionKind {
            and_gate,
            or_gate,
            at_least,
        };

        struct Definition final {
            std::string name;
            DefinitionKind kind = DefinitionKind::or_gate;
            std::uint32_t threshold = 0;
            std::vector<std::string> operands;
            std::size_t line = 0;
        };

        struct ParsedModel final {
            std::vector<Definition> definitions;
            std::unordered_map<std::string, std::size_t> definitionByName;
            std::unordered_map<std::string, double> probabilities;
            std::vector<std::string> probabilityNames;
            std::vector<std::string> explicitTargets;
            std::unordered_set<std::string> referencedNames;
            read_metadata metadata;
            api::diagnostic_list diagnostics;
        };

        [[nodiscard]] auto trim(std::string_view text) -> std::string_view {
            while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
                text.remove_prefix(1);
            }
            while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
                text.remove_suffix(1);
            }
            return text;
        }

        [[nodiscard]] auto splitWhitespace(std::string_view text) -> std::vector<std::string> {
            std::vector<std::string> tokens;
            while (!text.empty()) {
                text = trim(text);
                if (text.empty()) {
                    break;
                }
                auto end = std::size_t{0};
                while (end < text.size() && std::isspace(static_cast<unsigned char>(text[end])) == 0) {
                    ++end;
                }
                tokens.emplace_back(text.substr(0, end));
                text.remove_prefix(end);
            }
            return tokens;
        }

        [[nodiscard]] auto parseDouble(std::string_view token) -> std::optional<double> {
            std::string owned(token);
            char *end = nullptr;
            const auto value = std::strtod(owned.c_str(), &end);
            if (end == owned.c_str() || *end != '\0') {
                return std::nullopt;
            }
            return value;
        }

        [[nodiscard]] auto parseUnsigned(std::string_view token) -> std::optional<std::uint32_t> {
            std::uint32_t value = 0;
            const auto begin = token.data();
            const auto end = token.data() + token.size();
            const auto [cursor, error] = std::from_chars(begin, end, value);
            if (error != std::errc{} || cursor != end) {
                return std::nullopt;
            }
            return value;
        }

        [[nodiscard]] auto isGeneratedTargetName(std::string_view token) -> bool {
            if (token.size() < 2 || token.front() != 'T') {
                return false;
            }
            token.remove_prefix(1);
            return std::ranges::all_of(token, [](const char character) {
                return std::isdigit(static_cast<unsigned char>(character)) != 0;
            });
        }

        void addDefinition(ParsedModel &model, Definition definition) {
            for (auto const &operand : definition.operands) {
                model.referencedNames.insert(operand);
            }
            const auto [position, inserted] = model.definitionByName.emplace(definition.name, model.definitions.size());
            if (!inserted) {
                model.diagnostics.push(api::make_error("io.ftp.duplicate-definition",
                                                       "duplicate gate definition for '" + definition.name + "'"));
                return;
            }
            (void)position;
            model.definitions.push_back(std::move(definition));
        }

        void appendOperandsToPreviousDefinition(ParsedModel &model, std::vector<std::string> const &operands) {
            if (model.definitions.empty()) {
                return;
            }
            auto &definition = model.definitions.back();
            for (auto const &operand : operands) {
                definition.operands.push_back(operand);
                model.referencedNames.insert(operand);
            }
        }

        void parseLine(std::string_view line, std::size_t lineNumber, ParsedModel &model) {
            const auto isIndented = !line.empty() && std::isspace(static_cast<unsigned char>(line.front())) != 0;
            line = trim(line);
            if (line.empty() || line == "ENDTREE") {
                return;
            }
            if (line.front() == '*') {
                return;
            }

            const auto tokens = splitWhitespace(line);
            if (tokens.empty()) {
                return;
            }
            if (tokens[0] == "LIMIT") {
                if (tokens.size() < 2) {
                    model.diagnostics.push(api::make_error("io.ftp.invalid-limit", "LIMIT requires a numeric value"));
                    return;
                }
                if (const auto value = parseDouble(tokens[1])) {
                    model.metadata.truncation_limit = *value;
                    return;
                }
                model.diagnostics.push(api::make_error("io.ftp.invalid-limit", "LIMIT value is not numeric"));
                return;
            }
            if (tokens[0] == "PROCESS" || tokens[0] == "IMPORT") {
                return;
            }
            if (tokens.size() == 3 && tokens[1] == "+" && isGeneratedTargetName(tokens[0])) {
                model.explicitTargets.push_back(tokens[2]);
                return;
            }

            if (const auto probability = parseDouble(tokens[0]); probability && tokens.size() >= 2) {
                if (!model.probabilities.contains(tokens[1])) {
                    model.probabilityNames.push_back(tokens[1]);
                }
                model.probabilities.insert_or_assign(tokens[1], *probability);
                return;
            }

            if (isIndented) {
                appendOperandsToPreviousDefinition(model, tokens);
                return;
            }

            if (tokens.size() < 3) {
                model.diagnostics.push(api::make_warning("io.ftp.ignored-line", "ignored non-FTP line " + std::to_string(lineNumber)));
                return;
            }

            Definition definition;
            definition.name = tokens[0];
            definition.line = lineNumber;
            definition.operands.assign(tokens.begin() + 2, tokens.end());
            if (tokens[1] == "+") {
                definition.kind = DefinitionKind::or_gate;
            } else if (tokens[1] == "*") {
                definition.kind = DefinitionKind::and_gate;
            } else if (const auto threshold = parseUnsigned(tokens[1])) {
                definition.kind = DefinitionKind::at_least;
                definition.threshold = *threshold;
            } else {
                model.diagnostics.push(api::make_error("io.ftp.unsupported-gate",
                                                       "unsupported FTP gate operator '" + tokens[1] + "' on line " +
                                                           std::to_string(lineNumber)));
                return;
            }
            addDefinition(model, std::move(definition));
        }

        [[nodiscard]] auto parse(std::istream &input) -> ParsedModel {
            ParsedModel model;
            std::string line;
            for (std::size_t lineNumber = 1; std::getline(input, line); ++lineNumber) {
                parseLine(line, lineNumber, model);
            }
            return model;
        }

        class PcBuilder final {
        public:
            explicit PcBuilder(ParsedModel const &model) : model_(model) {}

            [[nodiscard]] auto build(std::vector<std::string> const &targets) -> ir::pc::finalize_result {
                for (auto const &target : targets) {
                    const auto node = buildNode(target);
                    if (node.valid()) {
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
                for (auto const &operandName : definition.operands) {
                    const auto operand = buildNode(operandName);
                    if (!operand.valid()) {
                        visiting_.erase(definition.name);
                        return ir::pc::bool_node_id::invalid();
                    }
                    operands.push_back(ir::pc::edge_ref{.source = operand});
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

        [[nodiscard]] auto selectedTargets(ParsedModel const &model, api::option_set const &options) -> std::vector<std::string> {
            if (auto requested = options.get<std::string>(target_option()); requested && !requested->empty()) {
                return {*requested};
            }
            if (!model.explicitTargets.empty()) {
                return model.explicitTargets;
            }

            std::vector<std::string> targets;
            for (auto const &definition : model.definitions) {
                if (!model.referencedNames.contains(definition.name)) {
                    targets.push_back(definition.name);
                }
            }
            if (targets.empty() && !model.definitions.empty()) {
                targets.push_back(model.definitions.front().name);
            }
            if (targets.empty() && !model.probabilityNames.empty()) {
                targets = model.probabilityNames;
                std::ranges::sort(targets);
            }
            return targets;
        }

    } // namespace

    auto reader_endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("io.fuzzybits.ftp.reader");
    }

    auto truncation_limit_option() -> api::option_key {
        return api::option_key("io.ftp.truncation_limit");
    }

    auto target_option() -> api::option_key {
        return api::option_key("io.ftp.target");
    }

    auto reader_descriptor() -> io::reader_descriptor {
        return io::reader_descriptor{
            .name = "FuzzyBits FTP reader",
            .provider = api::provider_id("fuzzybits"),
            .endpoint = reader_endpoint_id(),
            .input_format = formats::ftp(),
            .produces =
                api::capability_set{
                    api::capability_id("ir.probabilistic_boolean_dag"),
                    api::capability_id("ir.explicit_edge_polarity"),
                    api::capability_id("ir.targets"),
                    api::capability_id("ir.literal_probabilities"),
                    api::capability_id("ir.parameter_probabilities"),
                },
            .options =
                api::option_schema{.entries = {
                                       api::option_spec{
                                           .key = truncation_limit_option(),
                                           .name = "Truncation limit",
                                           .default_value = std::monostate{},
                                           .help = "Optional reader-side cut/truncation limit override for FTP files.",
                                       },
                                       api::option_spec{
                                           .key = target_option(),
                                           .name = "Target",
                                           .default_value = std::string{},
                                           .help = "Optional FTP symbol to import as the sole PC target.",
                                       },
                                   }},
        };
    }

    auto read(std::istream &input) -> ftp_read_result {
        api::option_set options;
        return read(input, options);
    }

    auto read(std::istream &input, api::option_set const &options) -> ftp_read_result {
        auto model = parse(input);
        if (auto overrideLimit = options.get<double>(truncation_limit_option())) {
            model.metadata.truncation_limit = *overrideLimit;
        }

        const auto targets = selectedTargets(model, options);
        if (targets.empty()) {
            model.diagnostics.push(api::make_error("io.ftp.no-targets", "FTP input did not contain an importable target gate"));
            return ftp_read_result::fail(std::move(model.diagnostics));
        }

        auto finalized = PcBuilder(model).build(targets);
        model.diagnostics.append(std::move(finalized.diagnostics));
        if (model.diagnostics.has_errors() || !finalized.output.valid()) {
            return ftp_read_result::fail(std::move(model.diagnostics));
        }

        return ftp_read_result::ok(
            read_output{
                .model = finalized.output.to_handle(),
                .metadata = std::move(model.metadata),
            },
            std::move(model.diagnostics));
    }

    auto read(io::read_request const &request) -> io::read_result {
        if (!request.format.empty() && request.format != formats::ftp()) {
            return io::read_result::fail(api::make_error("io.ftp.format-mismatch", "FTP reader received a non-FTP format request"));
        }

        std::ifstream input(request.path);
        if (!input) {
            return io::read_result::fail(api::make_error("io.ftp.open-failed", "failed to open FTP input file"));
        }

        auto result = read(input, request.options);
        if (!result) {
            return io::read_result::fail(result.diagnostics());
        }
        auto diagnostics = result.diagnostics();
        auto model = std::move(result).value().model;
        return io::read_result::ok(std::move(model), std::move(diagnostics));
    }

} // namespace fuzzybits::io::backend::fuzzybits::ftp
