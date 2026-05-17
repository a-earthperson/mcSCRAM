#include "io/ftp/descriptor.h"

#include "api/capabilities.h"
#include "api/diagnostics.h"
#include "api/ids.h"
#include "api/options.h"
#include "api/result.h"
#include "io/descriptor.h"
#include "io/formats.h"
#include "ir/handle.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"

#include <cstdint>
#include <iomanip>
#include <ostream>
#include <string>
#include <variant>

namespace fuzzybits::io::backend::fuzzybits::ftp {
    namespace {

        [[nodiscard]] auto nodeName(ir::pc::circuit const &model, const ir::pc::bool_node_id node) -> std::string {
            const auto code = model.op_codes()[node.raw()];
            if (code == ir::pc::op_code::basic_event || code == ir::pc::op_code::constant || code == ir::pc::op_code::house_event) {
                return "E" + std::to_string(node.raw());
            }
            return "N" + std::to_string(node.raw());
        }

        [[nodiscard]] auto validateSerializable(ir::pc::circuit const &model) -> api::diagnostic_list {
            api::diagnostic_list diagnostics;
            for (std::uint32_t node = 0; node < model.node_count(); ++node) {
                const auto id = ir::pc::bool_node_id(node);
                const auto code = model.op_codes()[node];
                for (auto const &edge : model.operands(id)) {
                    if (edge.complement) {
                        diagnostics.push(api::make_error("io.ftp.unsupported-complement",
                                                         "FTP writer cannot serialize complemented PC edges"));
                        break;
                    }
                }

                switch (code) {
                case ir::pc::op_code::basic_event: {
                    const auto source = model.probability_source_refs()[node];
                    if (!source.valid() || source.raw() >= model.probability_sources().size() ||
                        model.probability_sources()[source.raw()].kind != ir::pc::probability_source_kind::literal) {
                        diagnostics.push(api::make_error("io.ftp.unsupported-probability",
                                                         "FTP writer requires literal probabilities for basic events"));
                    }
                    break;
                }
                case ir::pc::op_code::constant:
                case ir::pc::op_code::house_event:
                case ir::pc::op_code::and_gate:
                case ir::pc::op_code::or_gate:
                    break;
                case ir::pc::op_code::cardinality:
                    if (model.cardinality_high()[node] != model.operand_counts()[node]) {
                        diagnostics.push(api::make_error("io.ftp.unsupported-cardinality",
                                                         "FTP writer can serialize only at-least-k cardinality gates"));
                    }
                    break;
                case ir::pc::op_code::xor_gate:
                case ir::pc::op_code::not_gate:
                case ir::pc::op_code::nand_gate:
                case ir::pc::op_code::nor_gate:
                case ir::pc::op_code::xnor_gate:
                    diagnostics.push(api::make_error("io.ftp.unsupported-opcode",
                                                     "FTP writer cannot serialize this PC opcode without a lossy transform"));
                    break;
                }
            }
            return diagnostics;
        }

        void writeGateLine(ir::pc::circuit const &model, const ir::pc::bool_node_id node, std::ostream &output) {
            const auto code = model.op_codes()[node.raw()];
            if (code == ir::pc::op_code::basic_event || code == ir::pc::op_code::constant || code == ir::pc::op_code::house_event) {
                return;
            }

            output << nodeName(model, node) << ' ';
            if (code == ir::pc::op_code::and_gate) {
                output << '*';
            } else if (code == ir::pc::op_code::or_gate) {
                output << '+';
            } else {
                output << model.cardinality_low()[node.raw()];
            }

            for (auto const &edge : model.operands(node)) {
                output << ' ' << nodeName(model, edge.source);
            }
            output << '\n';
        }

        void writeProbabilityLine(ir::pc::circuit const &model, const ir::pc::bool_node_id node, std::ostream &output) {
            const auto code = model.op_codes()[node.raw()];
            if (code == ir::pc::op_code::constant || code == ir::pc::op_code::house_event) {
                output << std::scientific << std::setprecision(6) << (model.constant_values()[node.raw()] != 0 ? 1.0 : 0.0) << ' '
                       << nodeName(model, node) << '\n';
                return;
            }
            if (code != ir::pc::op_code::basic_event) {
                return;
            }

            const auto source = model.probability_source_refs()[node.raw()];
            output << std::scientific << std::setprecision(6) << model.probability_sources()[source.raw()].literal << ' '
                   << nodeName(model, node) << '\n';
        }

    } // namespace

    auto writer_endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("io.fuzzybits.ftp.writer");
    }

    auto writer_descriptor() -> io::writer_descriptor {
        return io::writer_descriptor{
            .name = "FuzzyBits FTP writer",
            .provider = api::provider_id("fuzzybits"),
            .endpoint = writer_endpoint_id(),
            .output_format = formats::ftp(),
            .accepts =
                api::capability_set{
                    api::capability_id("ir.probabilistic_boolean_dag"),
                    api::capability_id("ir.explicit_edge_polarity"),
                    api::capability_id("ir.targets"),
                    api::capability_id("ir.literal_probabilities"),
                },
            .options =
                api::option_schema{.entries = {
                                       api::option_spec{
                                           .key = truncation_limit_option(),
                                           .name = "Truncation limit",
                                           .default_value = std::monostate{},
                                           .help = "Optional LIMIT value to emit into the FTP trailer.",
                                       },
                                   }},
        };
    }

    auto write(ir::handle const &model, std::ostream &output) -> io::write_result {
        return write(model, output, io::write_request{.format = formats::ftp()});
    }

    auto write(ir::handle const &model, std::ostream &output, io::write_request const &request) -> io::write_result {
        if (!request.format.empty() && request.format != formats::ftp()) {
            return io::write_result::fail(api::make_error("io.ftp.format-mismatch", "FTP writer received a non-FTP format request"));
        }

        auto const *pc = ir::pc::circuit_from(model);
        if (pc == nullptr) {
            return io::write_result::fail(api::make_error("io.ftp.unsupported-ir", "FTP writer requires a PC IR handle"));
        }
        return write(*pc, output, request);
    }

    auto write(ir::pc::circuit const &model, std::ostream &output) -> io::write_result {
        return write(model, output, io::write_request{.format = formats::ftp()});
    }

    auto write(ir::pc::circuit const &model, std::ostream &output, io::write_request const &request) -> io::write_result {
        if (!request.format.empty() && request.format != formats::ftp()) {
            return io::write_result::fail(api::make_error("io.ftp.format-mismatch", "FTP writer received a non-FTP format request"));
        }

        auto diagnostics = validateSerializable(model);
        if (diagnostics.has_errors()) {
            return io::write_result::fail(std::move(diagnostics));
        }

        output << "**CHAR32\n";
        for (auto const &target : model.targets()) {
            output << "T" << target.id.raw() << " + " << nodeName(model, target.node) << '\n';
        }
        for (std::uint32_t node = 0; node < model.node_count(); ++node) {
            writeGateLine(model, ir::pc::bool_node_id(node), output);
        }
        for (std::uint32_t node = 0; node < model.node_count(); ++node) {
            writeProbabilityLine(model, ir::pc::bool_node_id(node), output);
        }
        if (auto limit = request.options.get<double>(truncation_limit_option())) {
            output << "LIMIT    " << std::scientific << std::setprecision(6) << *limit << '\n';
        }
        output << "**FTITLE:\n";
        output << "*XEQ\n";
        return io::write_result::ok(true);
    }

} // namespace fuzzybits::io::backend::fuzzybits::ftp
