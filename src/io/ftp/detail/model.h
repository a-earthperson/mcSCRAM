#pragma once

#include "api/diagnostics.h"
#include "io/ftp/descriptor.h"

#include <cstddef>
#include <cstdint>
#include <istream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {

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

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
