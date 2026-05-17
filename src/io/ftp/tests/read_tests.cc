#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "io/ftp/descriptor.h"
#include "io/formats.h"
#include "ir/pc/circuit.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"

#include <cmath>
#include <sstream>

TEST_CASE("FTP reader descriptor advertises the FTP format", "[io][ftp]") {
    const auto descriptor = fuzzybits::io::backend::fuzzybits::ftp::reader_descriptor();

    REQUIRE(descriptor.provider == fuzzybits::api::provider_id("fuzzybits"));
    REQUIRE(descriptor.endpoint == fuzzybits::io::backend::fuzzybits::ftp::reader_endpoint_id());
    REQUIRE(descriptor.input_format == fuzzybits::io::formats::ftp());
}

TEST_CASE("FTP reader imports a simple circuit", "[io][ftp]") {
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

    REQUIRE(result);
    REQUIRE(result.value().metadata.truncation_limit);
    REQUIRE(std::abs(*result.value().metadata.truncation_limit - 1.0e-5) <= 1.0e-12);

    auto const *pc = fuzzybits::ir::pc::circuit_from(result.value().model);
    REQUIRE(pc != nullptr);
    REQUIRE(pc->node_count() == 5);
    REQUIRE(pc->edge_count() == 4);
    REQUIRE(pc->targets().size() == 1);
    REQUIRE(pc->probability_sources().size() == 3);
    REQUIRE(pc->probability_sources()[0].kind == fuzzybits::ir::pc::probability_source_kind::literal);
    REQUIRE(pc->op_codes()[pc->targets()[0].node.raw()] == fuzzybits::ir::pc::op_code::or_gate);
}
