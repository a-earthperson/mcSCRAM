#include "io/aiger/descriptor.h"
#include "io/aiger/detail/model.h"

#include "api/capabilities.h"
#include "api/diagnostics.h"
#include "api/ids.h"
#include "api/options.h"
#include "io/descriptor.h"
#include "io/formats.h"
#include "ir/handle.h"
#include "ir/pc/circuit.h"

#include <ostream>
#include <variant>

namespace fuzzybits::io::backend::fuzzybits::aiger {

    auto writer_endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("io.fuzzybits.aiger.writer");
    }

    auto writer_descriptor() -> io::writer_descriptor {
        return io::writer_descriptor{
            .name = "FuzzyBits AIGER writer",
            .provider = api::provider_id("fuzzybits"),
            .endpoint = writer_endpoint_id(),
            .output_format = formats::aiger(),
            .accepts =
                api::capability_set{
                    api::capability_id("ir.probabilistic_boolean_dag"),
                    api::capability_id("ir.explicit_edge_polarity"),
                    api::capability_id("ir.targets"),
                    api::capability_id("ir.parameter_probabilities"),
                },
            .options = api::option_schema{},
        };
    }

    auto write(ir::handle const &model, std::ostream &output) -> io::write_result {
        return write(model, output, io::write_request{.format = formats::aiger()});
    }

    auto write(ir::handle const &model, std::ostream &output, io::write_request const &request) -> io::write_result {
        if (!request.format.empty() && request.format != formats::aiger()) {
            return io::write_result::fail(api::make_error("io.aiger.format-mismatch", "AIGER writer received a non-AIGER format request"));
        }

        auto const *pc = ir::pc::circuit_from(model);
        if (pc == nullptr) {
            return io::write_result::fail(api::make_error("io.aiger.unsupported-ir", "AIGER writer requires a PC IR handle"));
        }
        return detail::write_pc(*pc, output, request);
    }

} // namespace fuzzybits::io::backend::fuzzybits::aiger
