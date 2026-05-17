#pragma once

#include "io/aiger/detail/model.h"

#include <iosfwd>

namespace fuzzybits::io::backend::fuzzybits::aiger::detail {

    [[nodiscard]] auto parse(std::istream &input) -> ParsedModel;

} // namespace fuzzybits::io::backend::fuzzybits::aiger::detail
