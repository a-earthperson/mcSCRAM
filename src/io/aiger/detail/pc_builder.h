#pragma once

#include "io/aiger/detail/model.h"
#include "ir/pc/builder.h"

namespace fuzzybits::io::backend::fuzzybits::aiger::detail {

    [[nodiscard]] auto build_pc(ParsedModel const &model) -> ir::pc::finalize_result;

} // namespace fuzzybits::io::backend::fuzzybits::aiger::detail
