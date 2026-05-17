#include "io/aiger/detail/parser.h"

#include "api/diagnostics.h"
#include "ir/pc/id.h"

#include <charconv>
#include <cctype>
#include <cstdint>
#include <istream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::aiger::detail {
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

        [[nodiscard]] auto splitWhitespace(std::string_view text) -> std::vector<std::string_view> {
            std::vector<std::string_view> tokens;
            while (!text.empty()) {
                text = trim(text);
                if (text.empty()) {
                    break;
                }
                auto end = std::size_t{0};
                while (end < text.size() && std::isspace(static_cast<unsigned char>(text[end])) == 0) {
                    ++end;
                }
                tokens.push_back(text.substr(0, end));
                text.remove_prefix(end);
            }
            return tokens;
        }

        [[nodiscard]] auto parseUnsigned(std::string_view token) -> std::optional<std::uint32_t> {
            if (token.empty()) {
                return std::nullopt;
            }
            std::uint64_t value = 0;
            const auto begin = token.data();
            const auto end = token.data() + token.size();
            const auto [cursor, error] = std::from_chars(begin, end, value);
            if (error != std::errc{} || cursor != end || value > std::numeric_limits<std::uint32_t>::max()) {
                return std::nullopt;
            }
            return static_cast<std::uint32_t>(value);
        }

        [[nodiscard]] auto checkedLiteralLimit(Header const &header) -> bool {
            return header.maxVariable <= ((std::numeric_limits<std::uint32_t>::max() - 1U) >> 1U);
        }

        [[nodiscard]] auto parseHeader(const std::string_view line, api::diagnostic_list &diagnostics) -> std::optional<Header> {
            const auto tokens = splitWhitespace(line);
            if (tokens.size() < 6) {
                diagnostics.push(api::make_error("io.aiger.invalid-header", "AIGER header must contain format and MILOA counts"));
                return std::nullopt;
            }

            Header header;
            if (tokens[0] == "aag") {
                header.encoding = Encoding::ascii;
            } else if (tokens[0] == "aig") {
                header.encoding = Encoding::binary;
            } else {
                diagnostics.push(api::make_error("io.aiger.invalid-header", "AIGER header must start with 'aag' or 'aig'"));
                return std::nullopt;
            }

            auto const maxVariable = parseUnsigned(tokens[1]);
            auto const inputCount = parseUnsigned(tokens[2]);
            auto const latchCount = parseUnsigned(tokens[3]);
            auto const outputCount = parseUnsigned(tokens[4]);
            auto const andCount = parseUnsigned(tokens[5]);
            if (!maxVariable || !inputCount || !latchCount || !outputCount || !andCount) {
                diagnostics.push(api::make_error("io.aiger.invalid-header", "AIGER MILOA counts must be unsigned 32-bit integers"));
                return std::nullopt;
            }

            header.maxVariable = *maxVariable;
            header.inputCount = *inputCount;
            header.latchCount = *latchCount;
            header.outputCount = *outputCount;
            header.andCount = *andCount;
            // This endpoint intentionally implements the original MILOA AIGER
            // circuit core only. Extension sections, even when standardized by
            // later AIGER revisions, can carry properties/probes/constraints that
            // have no lossless PC target/projection representation here.
            for (auto extra = tokens.begin() + 6; extra != tokens.end(); ++extra) {
                auto const count = parseUnsigned(*extra);
                if (!count) {
                    diagnostics.push(api::make_error("io.aiger.invalid-header", "AIGER extension counts must be unsigned 32-bit integers"));
                    return std::nullopt;
                }
                if (*count != 0) {
                    diagnostics.push(api::make_error("io.aiger.unsupported-extensions",
                                                     "AIGER reader supports only MILOA combinational files without extension sections"));
                    return std::nullopt;
                }
            }
            if (!checkedLiteralLimit(header)) {
                diagnostics.push(api::make_error("io.aiger.literal-overflow", "AIGER literals exceed the PC reader's 32-bit literal range"));
                return std::nullopt;
            }
            if (header.latchCount != 0) {
                // Sequential state would require latch semantics, initialization,
                // and next-state projection. The current PC IR is a flat
                // combinational Boolean DAG, so accepting latches would either
                // drop behavior or silently compile a transition system.
                diagnostics.push(api::make_error("io.aiger.unsupported-latches", "AIGER reader supports only combinational AIGER files"));
                return std::nullopt;
            }
            // Complemented output literals may become explicit PC NOT nodes because
            // targets point at nodes, not edge_ref values. The +outputCount bound
            // keeps that worst-case materialization inside the 32-bit PC node id
            // range before allocating builder storage.
            const auto maxPcNodes = static_cast<std::uint64_t>(std::numeric_limits<ir::pc::bool_node_id::value_type>::max());
            const auto worstCasePcNodes = static_cast<std::uint64_t>(header.inputCount) + header.andCount + header.outputCount + 2U;
            if (worstCasePcNodes > maxPcNodes) {
                diagnostics.push(api::make_error("io.aiger.too-large", "AIGER input exceeds the PC reader's 32-bit node range"));
                return std::nullopt;
            }
            if (header.encoding == Encoding::binary && header.maxVariable != header.inputCount + header.andCount) {
                diagnostics.push(api::make_error("io.aiger.invalid-binary-header", "binary AIGER requires M == I + L + A"));
                return std::nullopt;
            }
            if (header.inputCount > header.maxVariable) {
                diagnostics.push(api::make_error("io.aiger.invalid-header", "AIGER input count exceeds M"));
                return std::nullopt;
            }
            return header;
        }

        [[nodiscard]] auto readUnsignedLine(std::istream &input, std::string_view section, api::diagnostic_list &diagnostics)
            -> std::optional<std::uint32_t> {
            std::string line;
            if (!std::getline(input, line)) {
                diagnostics.push(api::make_error("io.aiger.unexpected-eof", "unexpected end of file while reading AIGER " + std::string(section)));
                return std::nullopt;
            }
            auto const value = parseUnsigned(trim(line));
            if (!value) {
                diagnostics.push(api::make_error("io.aiger.invalid-integer", "invalid unsigned integer in AIGER " + std::string(section)));
                return std::nullopt;
            }
            return value;
        }

        [[nodiscard]] auto readAsciiAndLine(std::istream &input, api::diagnostic_list &diagnostics) -> std::optional<AndDefinition> {
            std::string line;
            if (!std::getline(input, line)) {
                diagnostics.push(api::make_error("io.aiger.unexpected-eof", "unexpected end of file while reading AIGER AND gates"));
                return std::nullopt;
            }
            const auto tokens = splitWhitespace(line);
            if (tokens.size() != 3) {
                diagnostics.push(api::make_error("io.aiger.invalid-and", "ASCII AIGER AND definition must contain lhs rhs0 rhs1"));
                return std::nullopt;
            }
            auto const lhs = parseUnsigned(tokens[0]);
            auto const rhs0 = parseUnsigned(tokens[1]);
            auto const rhs1 = parseUnsigned(tokens[2]);
            if (!lhs || !rhs0 || !rhs1) {
                diagnostics.push(api::make_error("io.aiger.invalid-and", "ASCII AIGER AND literals must be unsigned integers"));
                return std::nullopt;
            }
            return AndDefinition{
                .lhs = *lhs,
                .rhs = {*rhs0, *rhs1},
            };
        }

        [[nodiscard]] auto decodeDelta(std::istream &input, api::diagnostic_list &diagnostics) -> std::optional<std::uint32_t> {
            std::uint64_t value = 0;
            auto shift = std::uint32_t{0};
            while (true) {
                const auto next = input.get();
                if (next == std::char_traits<char>::eof()) {
                    diagnostics.push(api::make_error("io.aiger.unexpected-eof", "unexpected end of file in binary AIGER delta"));
                    return std::nullopt;
                }
                const auto byte = static_cast<unsigned char>(next);
                value |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
                if ((byte & 0x80U) == 0) {
                    break;
                }
                shift += 7;
                if (shift >= 35 || value > std::numeric_limits<std::uint32_t>::max()) {
                    diagnostics.push(api::make_error("io.aiger.delta-overflow", "binary AIGER delta exceeds the 32-bit literal range"));
                    return std::nullopt;
                }
            }
            if (value > std::numeric_limits<std::uint32_t>::max()) {
                diagnostics.push(api::make_error("io.aiger.delta-overflow", "binary AIGER delta exceeds the 32-bit literal range"));
                return std::nullopt;
            }
            return static_cast<std::uint32_t>(value);
        }

        void parseSymbolLine(std::string_view line, ParsedModel &model, std::unordered_set<std::uint32_t> &seenInputSymbols) {
            if (line.empty()) {
                return;
            }
            if (line.front() == 'c') {
                return;
            }
            if (line.front() != 'i' && line.front() != 'l' && line.front() != 'o') {
                model.diagnostics.push(api::make_error("io.aiger.invalid-symbol", "AIGER symbol line has an invalid symbol kind"));
                return;
            }

            const auto space = line.find(' ');
            if (space == std::string_view::npos) {
                model.diagnostics.push(api::make_error("io.aiger.invalid-symbol", "AIGER symbol line must contain a symbol name"));
                return;
            }
            auto const position = parseUnsigned(line.substr(1, space - 1));
            if (!position) {
                model.diagnostics.push(api::make_error("io.aiger.invalid-symbol", "AIGER symbol position must be an unsigned integer"));
                return;
            }
            if (line.front() != 'i') {
                return;
            }
            if (*position >= model.header.inputCount) {
                model.diagnostics.push(api::make_error("io.aiger.invalid-symbol", "AIGER input symbol index is out of range"));
                return;
            }
            if (!seenInputSymbols.insert(*position).second) {
                model.diagnostics.push(api::make_error("io.aiger.duplicate-symbol", "AIGER input symbol is defined more than once"));
                return;
            }
            model.inputSymbols[*position] = std::string(line.substr(space + 1));
        }

        void parseTrailingSymbols(std::istream &input, ParsedModel &model) {
            std::unordered_set<std::uint32_t> seenInputSymbols;
            std::string line;
            while (std::getline(input, line)) {
                if (line == "c") {
                    break;
                }
                parseSymbolLine(line, model, seenInputSymbols);
            }
        }

        void validateLiteral(std::uint32_t literal, Header const &header, api::diagnostic_list &diagnostics, std::string_view context) {
            if (literalVariable(literal) > header.maxVariable) {
                diagnostics.push(api::make_error("io.aiger.literal-out-of-range",
                                                 "AIGER " + std::string(context) + " literal exceeds M"));
            }
        }

        void validateParsedModel(ParsedModel &model) {
            std::unordered_set<std::uint32_t> definedVariables;
            definedVariables.reserve(model.inputLiterals.size() + model.andDefinitions.size());

            for (auto const literal : model.inputLiterals) {
                validateLiteral(literal, model.header, model.diagnostics, "input");
                if (literal == 0 || (literal & 1U) != 0) {
                    model.diagnostics.push(api::make_error("io.aiger.invalid-input", "AIGER input literal must be positive and even"));
                    continue;
                }
                if (!definedVariables.insert(literalVariable(literal)).second) {
                    model.diagnostics.push(api::make_error("io.aiger.duplicate-definition", "AIGER variable is defined more than once"));
                }
            }

            for (auto const literal : model.outputLiterals) {
                validateLiteral(literal, model.header, model.diagnostics, "output");
            }

            for (auto const &definition : model.andDefinitions) {
                validateLiteral(definition.lhs, model.header, model.diagnostics, "AND lhs");
                if (definition.lhs == 0 || (definition.lhs & 1U) != 0) {
                    model.diagnostics.push(api::make_error("io.aiger.invalid-and", "AIGER AND lhs must be positive and even"));
                }
                if (!definedVariables.insert(literalVariable(definition.lhs)).second) {
                    model.diagnostics.push(api::make_error("io.aiger.duplicate-definition", "AIGER variable is defined more than once"));
                }
                for (auto const rhs : definition.rhs) {
                    validateLiteral(rhs, model.header, model.diagnostics, "AND rhs");
                }
            }

            for (auto const literal : model.outputLiterals) {
                const auto variable = literalVariable(literal);
                if (variable != 0 && !definedVariables.contains(variable)) {
                    model.diagnostics.push(api::make_error("io.aiger.undefined-literal", "AIGER output references an undefined literal"));
                }
            }
            for (auto const &definition : model.andDefinitions) {
                for (auto const rhs : definition.rhs) {
                    const auto variable = literalVariable(rhs);
                    if (variable != 0 && !definedVariables.contains(variable)) {
                        model.diagnostics.push(api::make_error("io.aiger.undefined-literal", "AIGER AND references an undefined literal"));
                    }
                }
            }
        }

        [[nodiscard]] auto parseAscii(std::istream &input, Header header, api::diagnostic_list diagnostics) -> ParsedModel {
            ParsedModel model{
                .header = header,
                .diagnostics = std::move(diagnostics),
            };
            model.inputLiterals.reserve(header.inputCount);
            model.outputLiterals.reserve(header.outputCount);
            model.andDefinitions.reserve(header.andCount);
            model.inputSymbols.resize(header.inputCount);

            for (std::uint32_t index = 0; index < header.inputCount; ++index) {
                if (auto literal = readUnsignedLine(input, "inputs", model.diagnostics)) {
                    model.inputLiterals.push_back(*literal);
                }
            }
            for (std::uint32_t index = 0; index < header.outputCount; ++index) {
                if (auto literal = readUnsignedLine(input, "outputs", model.diagnostics)) {
                    model.outputLiterals.push_back(*literal);
                }
            }
            for (std::uint32_t index = 0; index < header.andCount; ++index) {
                if (auto definition = readAsciiAndLine(input, model.diagnostics)) {
                    model.andDefinitions.push_back(*definition);
                }
            }
            parseTrailingSymbols(input, model);
            validateParsedModel(model);
            return model;
        }

        [[nodiscard]] auto parseBinary(std::istream &input, Header header, api::diagnostic_list diagnostics) -> ParsedModel {
            ParsedModel model{
                .header = header,
                .diagnostics = std::move(diagnostics),
            };
            model.inputLiterals.reserve(header.inputCount);
            model.outputLiterals.reserve(header.outputCount);
            model.andDefinitions.reserve(header.andCount);
            model.inputSymbols.resize(header.inputCount);

            for (std::uint32_t index = 0; index < header.inputCount; ++index) {
                // Binary AIGER omits input literals; by format definition they are
                // exactly 2, 4, ..., 2*I. This recreates file variables, not PC node
                // ids, which are assigned later by the builder.
                model.inputLiterals.push_back(2U * (index + 1U));
            }
            for (std::uint32_t index = 0; index < header.outputCount; ++index) {
                if (auto literal = readUnsignedLine(input, "outputs", model.diagnostics)) {
                    model.outputLiterals.push_back(*literal);
                }
            }
            for (std::uint32_t index = 0; index < header.andCount; ++index) {
                const auto lhs = 2U * (header.inputCount + index + 1U);
                const auto delta0 = decodeDelta(input, model.diagnostics);
                const auto delta1 = decodeDelta(input, model.diagnostics);
                if (!delta0 || !delta1) {
                    break;
                }
                if (*delta0 > lhs) {
                    model.diagnostics.push(api::make_error("io.aiger.invalid-delta", "binary AIGER delta0 exceeds lhs"));
                    continue;
                }
                const auto rhs0 = lhs - *delta0;
                if (*delta1 > rhs0) {
                    model.diagnostics.push(api::make_error("io.aiger.invalid-delta", "binary AIGER delta1 exceeds rhs0"));
                    continue;
                }
                const auto rhs1 = rhs0 - *delta1;
                if (!(lhs > rhs0 && rhs0 >= rhs1)) {
                    model.diagnostics.push(api::make_error("io.aiger.invalid-delta", "binary AIGER AND literals violate lhs > rhs0 >= rhs1"));
                    continue;
                }
                model.andDefinitions.push_back(AndDefinition{
                    .lhs = lhs,
                    .rhs = {rhs0, rhs1},
                });
            }
            parseTrailingSymbols(input, model);
            validateParsedModel(model);
            return model;
        }

    } // namespace

    auto parse(std::istream &input) -> ParsedModel {
        api::diagnostic_list diagnostics;
        std::string headerLine;
        if (!std::getline(input, headerLine)) {
            diagnostics.push(api::make_error("io.aiger.empty-input", "AIGER input is empty"));
            return ParsedModel{.diagnostics = std::move(diagnostics)};
        }
        auto header = parseHeader(headerLine, diagnostics);
        if (!header) {
            return ParsedModel{.diagnostics = std::move(diagnostics)};
        }
        if (header->encoding == Encoding::ascii) {
            return parseAscii(input, *header, std::move(diagnostics));
        }
        return parseBinary(input, *header, std::move(diagnostics));
    }

} // namespace fuzzybits::io::backend::fuzzybits::aiger::detail
