#pragma once

#include "io/descriptor.h"
#include "ir/pc/circuit.h"

#include <iosfwd>

namespace fuzzybits::io::backend::fuzzybits::aiger::detail {

    [[nodiscard]] auto write_pc(ir::pc::circuit const &model, std::ostream &output, io::write_request const &request)
        -> io::write_result;

} // namespace fuzzybits::io::backend::fuzzybits::aiger::detail
