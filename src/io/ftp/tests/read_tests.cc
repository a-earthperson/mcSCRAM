#include "io/ftp/descriptor.h"
#include "io/formats.h"
#include "ir/pc/circuit.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"

#include <cmath>
#include <sstream>

namespace {

    [[nodiscard]] auto readsSimpleFtpCircuit() -> bool {
        std::istringstream input(R"ftp(**CHAR32
TOP + A MID
MID * B C
         1.000000E-01 A
         2.000000E-01 B
         3.000000E-01 C
LIMIT    1.000000E-05
**FTITLE:
*XEQ
)ftp");

        auto result = fuzzybits::io::backend::fuzzybits::ftp::read(input);
        if (!result || !result.value().metadata.truncation_limit) {
            return false;
        }
        if (std::abs(*result.value().metadata.truncation_limit - 1.0e-5) > 1.0e-12) {
            return false;
        }

        auto const *pc = fuzzybits::ir::pc::circuit_from(result.value().model);
        if (pc == nullptr || pc->node_count() != 5 || pc->edge_count() != 4 || pc->targets().size() != 1) {
            return false;
        }
        return pc->probability_sources().size() == 3 &&
               pc->probability_sources()[0].kind == fuzzybits::ir::pc::probability_source_kind::literal &&
               pc->op_codes()[pc->targets()[0].node.raw()] == fuzzybits::ir::pc::op_code::or_gate;
    }

    [[nodiscard]] auto descriptorUsesFtpFormat() -> bool {
        const auto descriptor = fuzzybits::io::backend::fuzzybits::ftp::reader_descriptor();
        return descriptor.provider == fuzzybits::api::provider_id("fuzzybits") &&
               descriptor.endpoint == fuzzybits::io::backend::fuzzybits::ftp::reader_endpoint_id() &&
               descriptor.input_format == fuzzybits::io::formats::ftp();
    }

} // namespace

int main() {
    if (!descriptorUsesFtpFormat()) {
        return 1;
    }
    if (!readsSimpleFtpCircuit()) {
        return 2;
    }
    return 0;
}
