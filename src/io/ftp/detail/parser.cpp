#include "io/ftp/detail/parser.h"

#include "api/diagnostics.h"

#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {
    namespace {

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

        [[nodiscard]] auto parseOperand(std::string token) -> Operand {
            if (token.size() > 1 && token.front() == '-') {
                token.erase(token.begin());
                return Operand{
                    .name = std::move(token),
                    .complement = true,
                };
            }
            return Operand{
                .name = std::move(token),
            };
        }

        void addDefinition(ParsedModel &model, Definition definition) {
            for (auto const &operand : definition.operands) {
                model.referencedNames.insert(operand.name);
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
                auto parsed = parseOperand(operand);
                model.referencedNames.insert(parsed.name);
                definition.operands.push_back(std::move(parsed));
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
            definition.operands.reserve(tokens.size() - 2);
            for (auto operand = tokens.begin() + 2; operand != tokens.end(); ++operand) {
                definition.operands.push_back(parseOperand(*operand));
            }
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

    } // namespace

    auto parse(std::istream &input) -> ParsedModel {
        ParsedModel model;
        std::string line;
        for (std::size_t lineNumber = 1; std::getline(input, line); ++lineNumber) {
            parseLine(line, lineNumber, model);
        }
        return model;
    }

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
