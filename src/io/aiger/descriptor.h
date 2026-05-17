#pragma once

#include "api/ids.h"
#include "api/options.h"
#include "api/result.h"
#include "io/descriptor.h"
#include "ir/handle.h"
#include "ir/pc/circuit.h"

#include <iosfwd>

namespace fuzzybits::io::backend::fuzzybits::aiger {

    struct read_output final {
        ir::handle model;
    };

    using aiger_read_result = api::result<read_output>;

    [[nodiscard]] auto reader_endpoint_id() -> api::endpoint_id;
    [[nodiscard]] auto writer_endpoint_id() -> api::endpoint_id;
    [[nodiscard]] auto reader_descriptor() -> io::reader_descriptor;
    [[nodiscard]] auto writer_descriptor() -> io::writer_descriptor;

    [[nodiscard]] auto read(std::istream &input) -> aiger_read_result;
    [[nodiscard]] auto read(std::istream &input, api::option_set const &options) -> aiger_read_result;
    [[nodiscard]] auto read(io::read_request const &request) -> io::read_result;

    [[nodiscard]] auto write(ir::handle const &model, std::ostream &output) -> io::write_result;
    [[nodiscard]] auto write(ir::handle const &model, std::ostream &output, io::write_request const &request) -> io::write_result;
    [[nodiscard]] auto write(ir::pc::circuit const &model, std::ostream &output) -> io::write_result;
    [[nodiscard]] auto write(ir::pc::circuit const &model, std::ostream &output, io::write_request const &request) -> io::write_result;

} // namespace fuzzybits::io::backend::fuzzybits::aiger
