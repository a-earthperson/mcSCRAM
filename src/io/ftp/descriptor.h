#pragma once

#include "api/ids.h"
#include "api/options.h"
#include "api/result.h"
#include "io/descriptor.h"
#include "ir/handle.h"

#include <iosfwd>
#include <optional>

namespace fuzzybits::io::backend::fuzzybits::ftp {

    struct read_metadata final {
        std::optional<double> truncation_limit;
    };

    struct read_output final {
        ir::handle model;
        read_metadata metadata;
    };

    using ftp_read_result = api::result<read_output>;

    [[nodiscard]] auto reader_endpoint_id() -> api::endpoint_id;
    [[nodiscard]] auto writer_endpoint_id() -> api::endpoint_id;
    [[nodiscard]] auto truncation_limit_option() -> api::option_key;
    [[nodiscard]] auto target_option() -> api::option_key;
    [[nodiscard]] auto reader_descriptor() -> io::reader_descriptor;
    [[nodiscard]] auto writer_descriptor() -> io::writer_descriptor;

    [[nodiscard]] auto read(std::istream &input) -> ftp_read_result;
    [[nodiscard]] auto read(std::istream &input, api::option_set const &options) -> ftp_read_result;
    [[nodiscard]] auto read(io::read_request const &request) -> io::read_result;

    [[nodiscard]] auto write(ir::handle const &model, std::ostream &output) -> io::write_result;
    [[nodiscard]] auto write(ir::handle const &model, std::ostream &output, io::write_request const &request) -> io::write_result;

} // namespace fuzzybits::io::backend::fuzzybits::ftp
