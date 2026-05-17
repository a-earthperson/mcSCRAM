#include "io/ftp/descriptor.h"
#include "io/formats.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"

#include <sstream>
#include <string>

namespace {

    [[nodiscard]] auto roundTripsWritablePcSubset() -> bool {
        fuzzybits::ir::pc::builder builder;
        const auto aProbability = builder.add_literal_probability(0.1);
        const auto bProbability = builder.add_literal_probability(0.2);
        const auto a = builder.add_basic_event(aProbability);
        const auto b = builder.add_basic_event(bProbability);
        const auto top = builder.add_and({
            fuzzybits::ir::pc::edge_ref{.source = a},
            fuzzybits::ir::pc::edge_ref{.source = b},
        });
        const auto target = builder.add_target(top);
        (void)target;

        auto finalized = std::move(builder).seal();
        if (!finalized.ok()) {
            return false;
        }

        fuzzybits::io::write_request request{
            .format = fuzzybits::io::formats::ftp(),
        };
        request.options.set(fuzzybits::io::backend::fuzzybits::ftp::truncation_limit_option(), 1.0e-6);

        std::ostringstream output;
        auto written = fuzzybits::io::backend::fuzzybits::ftp::write(finalized.output, output, request);
        const auto text = output.str();
        if (!written || text.find("LIMIT") == std::string::npos || text.find("T0 + N") == std::string::npos) {
            return false;
        }

        std::istringstream input(text);
        auto read = fuzzybits::io::backend::fuzzybits::ftp::read(input);
        auto const *pc = read ? fuzzybits::ir::pc::circuit_from(read.value().model) : nullptr;
        return pc != nullptr && pc->targets().size() == 1 && pc->probability_sources().size() == 2;
    }

    [[nodiscard]] auto descriptorUsesFtpFormat() -> bool {
        const auto descriptor = fuzzybits::io::backend::fuzzybits::ftp::writer_descriptor();
        return descriptor.provider == fuzzybits::api::provider_id("fuzzybits") &&
               descriptor.endpoint == fuzzybits::io::backend::fuzzybits::ftp::writer_endpoint_id() &&
               descriptor.output_format == fuzzybits::io::formats::ftp();
    }

} // namespace

int main() {
    if (!descriptorUsesFtpFormat()) {
        return 1;
    }
    if (!roundTripsWritablePcSubset()) {
        return 2;
    }
    return 0;
}
