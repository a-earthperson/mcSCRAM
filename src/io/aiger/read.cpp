#include "io/aiger/descriptor.h"

#include "api/capabilities.h"
#include "api/diagnostics.h"
#include "api/ids.h"
#include "api/options.h"
#include "io/aiger/detail/parser.h"
#include "io/aiger/detail/pc_builder.h"
#include "io/descriptor.h"
#include "io/formats.h"

#include <fstream>
#include <istream>
#include <utility>
#include <variant>

namespace fuzzybits::io::backend::fuzzybits::aiger {

    auto reader_endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("io.fuzzybits.aiger.reader");
    }

    auto reader_descriptor() -> io::reader_descriptor {
        return io::reader_descriptor{
            .name = "FuzzyBits AIGER reader",
            .provider = api::provider_id("fuzzybits"),
            .endpoint = reader_endpoint_id(),
            .input_format = formats::aiger(),
            .produces =
                api::capability_set{
                    api::capability_id("ir.probabilistic_boolean_dag"),
                    api::capability_id("ir.explicit_edge_polarity"),
                    api::capability_id("ir.targets"),
                    api::capability_id("ir.parameter_probabilities"),
                },
            .options = api::option_schema{},
        };
    }

    auto read(std::istream &input) -> aiger_read_result {
        api::option_set options;
        return read(input, options);
    }

    auto read(std::istream &input, api::option_set const &options) -> aiger_read_result {
        (void)options;
        auto model = detail::parse(input);
        if (model.diagnostics.has_errors()) {
            return aiger_read_result::fail(std::move(model.diagnostics));
        }
        auto finalized = detail::build_pc(model);
        model.diagnostics.append(std::move(finalized.diagnostics));
        if (model.diagnostics.has_errors() || !finalized.output.valid()) {
            return aiger_read_result::fail(std::move(model.diagnostics));
        }

        return aiger_read_result::ok(read_output{.model = finalized.output.to_handle()}, std::move(model.diagnostics));
    }

    auto read(io::read_request const &request) -> io::read_result {
        if (!request.format.empty() && request.format != formats::aiger()) {
            return io::read_result::fail(api::make_error("io.aiger.format-mismatch", "AIGER reader received a non-AIGER format request"));
        }

        std::ifstream input(request.path, std::ios::binary);
        if (!input) {
            return io::read_result::fail(api::make_error("io.aiger.open-failed", "failed to open AIGER input file"));
        }

        auto result = read(input, request.options);
        if (!result) {
            return io::read_result::fail(result.diagnostics());
        }
        auto diagnostics = result.diagnostics();
        auto model = std::move(result).value().model;
        return io::read_result::ok(std::move(model), std::move(diagnostics));
    }

} // namespace fuzzybits::io::backend::fuzzybits::aiger
