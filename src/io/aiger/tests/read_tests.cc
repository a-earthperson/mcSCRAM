#include <catch2/catch.hpp>

#include "api/ids.h"
#include "api/diagnostics.h"
#include "io/aiger/descriptor.h"
#include "io/formats.h"
#include "ir/forms.h"
#include "ir/pc/circuit.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"

#include <sstream>
#include <string>

namespace {

    [[nodiscard]] auto diagnosticsText(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
        std::ostringstream output;
        for (auto const &diagnostic : diagnostics.items()) {
            output << diagnostic.code << ": " << diagnostic.message << '\n';
        }
        return output.str();
    }

} // namespace

TEST_CASE("AIGER reader descriptor advertises the AIGER format", "[io][aiger]") {
    const auto descriptor = fuzzybits::io::backend::fuzzybits::aiger::reader_descriptor();

    REQUIRE(descriptor.provider == fuzzybits::api::provider_id("fuzzybits"));
    REQUIRE(descriptor.endpoint == fuzzybits::io::backend::fuzzybits::aiger::reader_endpoint_id());
    REQUIRE(descriptor.input_format == fuzzybits::io::formats::aiger());
}

TEST_CASE("AIGER reader imports ASCII AIGER as AIG-shaped PC", "[io][aiger]") {
    std::istringstream input(R"aag(aag 3 2 0 1 1
2
4
7
6 2 5
i0 x
i1 y
o0 inverted_and
c
comment ignored
)aag");

    auto result = fuzzybits::io::backend::fuzzybits::aiger::read(input);
    INFO(diagnosticsText(result.diagnostics()));
    REQUIRE(result);

    auto const *pc = fuzzybits::ir::pc::circuit_from(result.value().model);
    REQUIRE(pc != nullptr);
    REQUIRE(pc->valid());
    REQUIRE(pc->form_profile().is_aig());
    REQUIRE(pc->satisfies_form(fuzzybits::ir::forms::aig()));
    REQUIRE(pc->targets().size() == 1);
    REQUIRE(pc->probability_sources().size() == 2);
    REQUIRE(pc->probability_sources()[0].parameter_name == "x");
    REQUIRE(pc->probability_sources()[1].parameter_name == "y");

    const auto target = pc->targets()[0].node;
    REQUIRE(pc->op_codes()[target.raw()] == fuzzybits::ir::pc::op_code::not_gate);
    const auto targetOperands = pc->operands(target);
    REQUIRE(targetOperands.size() == 1);
    REQUIRE(pc->op_codes()[targetOperands.front().source.raw()] == fuzzybits::ir::pc::op_code::and_gate);

    const auto andOperands = pc->operands(targetOperands.front().source);
    REQUIRE(andOperands.size() == 2);
    REQUIRE_FALSE(andOperands[0].complement);
    REQUIRE(andOperands[1].complement);
}

TEST_CASE("AIGER reader rejects sequential AIGER", "[io][aiger]") {
    std::istringstream input(R"aag(aag 1 0 1 1 0
2 3
2
)aag");

    auto result = fuzzybits::io::backend::fuzzybits::aiger::read(input);

    REQUIRE_FALSE(result);
    REQUIRE(result.diagnostics().has_errors());
}
