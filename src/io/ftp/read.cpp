#include "io/ftp/descriptor.h"
#include "io/ftp/detail/model.h"

#include "api/capabilities.h"
#include "api/diagnostics.h"
#include "api/ids.h"
#include "api/options.h"
#include "io/descriptor.h"
#include "io/formats.h"

#include <fstream>
#include <istream>
#include <string>
#include <utility>
#include <variant>

namespace fuzzybits::io::backend::fuzzybits::ftp {

    auto reader_endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("io.fuzzybits.ftp.reader");
    }

    auto truncation_limit_option() -> api::option_key {
        return api::option_key("io.ftp.truncation_limit");
    }

    auto target_option() -> api::option_key {
        return api::option_key("io.ftp.target");
    }

    auto reader_descriptor() -> io::reader_descriptor {
        return io::reader_descriptor{
            .name = "FuzzyBits FTP reader",
            .provider = api::provider_id("fuzzybits"),
            .endpoint = reader_endpoint_id(),
            .input_format = formats::ftp(),
            .produces =
                api::capability_set{
                    api::capability_id("ir.probabilistic_boolean_dag"),
                    api::capability_id("ir.explicit_edge_polarity"),
                    api::capability_id("ir.targets"),
                    api::capability_id("ir.literal_probabilities"),
                    api::capability_id("ir.parameter_probabilities"),
                },
            .options =
                api::option_schema{.entries = {
                                       api::option_spec{
                                           .key = truncation_limit_option(),
                                           .name = "Truncation limit",
                                           .default_value = std::monostate{},
                                           .help = "Optional reader-side cut/truncation limit override for FTP files.",
                                       },
                                       api::option_spec{
                                           .key = target_option(),
                                           .name = "Target",
                                           .default_value = std::string{},
                                           .help = "Optional FTP symbol to import as the sole PC target.",
                                       },
                                   }},
        };
    }

    auto read(std::istream &input) -> ftp_read_result {
        api::option_set options;
        return read(input, options);
    }

    auto read(std::istream &input, api::option_set const &options) -> ftp_read_result {
        auto model = detail::parse(input);
        if (auto overrideLimit = options.get<double>(truncation_limit_option())) {
            model.metadata.truncation_limit = *overrideLimit;
        }

        const auto targets = detail::selected_targets(model, options);
        if (targets.empty()) {
            model.diagnostics.push(api::make_error("io.ftp.no-targets", "FTP input did not contain an importable target gate"));
            return ftp_read_result::fail(std::move(model.diagnostics));
        }

        auto finalized = detail::build_pc(model, targets);
        model.diagnostics.append(std::move(finalized.diagnostics));
        if (model.diagnostics.has_errors() || !finalized.output.valid()) {
            return ftp_read_result::fail(std::move(model.diagnostics));
        }

        return ftp_read_result::ok(
            read_output{
                .model = finalized.output.to_handle(),
                .metadata = std::move(model.metadata),
            },
            std::move(model.diagnostics));
    }

    auto read(io::read_request const &request) -> io::read_result {
        if (!request.format.empty() && request.format != formats::ftp()) {
            return io::read_result::fail(api::make_error("io.ftp.format-mismatch", "FTP reader received a non-FTP format request"));
        }

        std::ifstream input(request.path);
        if (!input) {
            return io::read_result::fail(api::make_error("io.ftp.open-failed", "failed to open FTP input file"));
        }

        auto result = read(input, request.options);
        if (!result) {
            return io::read_result::fail(result.diagnostics());
        }
        auto diagnostics = result.diagnostics();
        auto model = std::move(result).value().model;
        return io::read_result::ok(std::move(model), std::move(diagnostics));
    }

} // namespace fuzzybits::io::backend::fuzzybits::ftp
