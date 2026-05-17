#pragma once

#include "api/diagnostics.h"
#include "api/options.h"
#include "io/ftp/descriptor.h"
#include "io/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {

    // On-wire FTP tokens shared by the parser and writer (single source of truth).
    inline constexpr char kFtpOrGateToken = '+';
    inline constexpr char kFtpAndGateToken = '*';
    inline constexpr char kFtpComplementToken = '-';
    inline constexpr char kFtpBasicEventNodePrefix = 'E';
    inline constexpr char kFtpCompositeNodePrefix = 'N';
    inline constexpr char kFtpGeneratedTargetPrefix = 'T';
    inline constexpr std::string_view kFtpEncodingLine = "**CHAR32\n";
    inline constexpr std::string_view kFtpFtitleLine = "**FTITLE:\n";
    inline constexpr std::string_view kFtpXeqLine = "*XEQ\n";
    inline constexpr std::string_view kFtpLimitOutputPrefix = "LIMIT    ";
    inline constexpr std::string_view kFtpEndTreeToken = "ENDTREE";
    inline constexpr std::string_view kFtpLimitToken = "LIMIT";
    inline constexpr std::string_view kFtpProcessToken = "PROCESS";
    inline constexpr std::string_view kFtpImportToken = "IMPORT";
    inline constexpr std::string_view kFtpTargetLineJoin = " + ";

    enum class DefinitionKind : std::uint8_t {
        and_gate,
        or_gate,
        at_least,
    };

    struct Operand final {
        std::string name;
        bool complement = false;
    };

    struct Definition final {
        std::string name;
        DefinitionKind kind = DefinitionKind::or_gate;
        std::uint32_t threshold = 0;
        std::vector<Operand> operands;
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

    [[nodiscard]] auto parse(std::istream &input) -> ParsedModel;
    [[nodiscard]] auto selected_targets(ParsedModel const &model, api::option_set const &options) -> std::vector<std::string>;
    [[nodiscard]] auto build_pc(ParsedModel const &model, std::vector<std::string> const &targets) -> ir::pc::finalize_result;
    [[nodiscard]] auto write_pc(ir::pc::circuit const &model, std::ostream &output, io::write_request const &request)
        -> io::write_result;

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
