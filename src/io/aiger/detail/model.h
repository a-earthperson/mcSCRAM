#pragma once

#include "api/diagnostics.h"
#include "io/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"

#include <array>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::aiger::detail {

    enum class Encoding : std::uint8_t {
        ascii,
        binary,
    };

    struct Header final {
        Encoding encoding = Encoding::ascii;
        std::uint32_t maxVariable = 0;
        std::uint32_t inputCount = 0;
        std::uint32_t latchCount = 0;
        std::uint32_t outputCount = 0;
        std::uint32_t andCount = 0;
    };

    struct AndDefinition final {
        std::uint32_t lhs = 0;
        std::array<std::uint32_t, 2> rhs{};
    };

    struct ParsedModel final {
        Header header;
        std::vector<std::uint32_t> inputLiterals;
        std::vector<std::uint32_t> outputLiterals;
        std::vector<AndDefinition> andDefinitions;
        // Only input symbols are imported into PC state, as parameter names for
        // basic-event probability sources. AIGER output/latch symbols and
        // comments are metadata without a current PC column, so preserving them
        // would require a separate projection/provenance contract.
        std::vector<std::string> inputSymbols;
        api::diagnostic_list diagnostics;
    };

    [[nodiscard]] inline auto literalVariable(const std::uint32_t literal) -> std::uint32_t {
        return literal >> 1U;
    }

    [[nodiscard]] auto parse(std::istream &input) -> ParsedModel;
    [[nodiscard]] auto build_pc(ParsedModel const &model) -> ir::pc::finalize_result;
    [[nodiscard]] auto write_pc(ir::pc::circuit const &model, std::ostream &output, io::write_request const &request)
    -> io::write_result;

} // namespace fuzzybits::io::backend::fuzzybits::aiger::detail
