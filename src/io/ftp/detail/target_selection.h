#pragma once

#include "api/options.h"
#include "io/ftp/detail/model.h"

#include <string>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {

    [[nodiscard]] auto selected_targets(ParsedModel const &model, api::option_set const &options) -> std::vector<std::string>;

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
