#pragma once

#include "io/ftp/detail/model.h"
#include "ir/pc/builder.h"

#include <string>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {

    [[nodiscard]] auto build_pc(ParsedModel const &model, std::vector<std::string> const &targets) -> ir::pc::finalize_result;

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
