#pragma once

#include "io/ftp/detail/model.h"

#include <iosfwd>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {

    [[nodiscard]] auto parse(std::istream &input) -> ParsedModel;

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
