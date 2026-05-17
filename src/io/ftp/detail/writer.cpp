#include "io/ftp/detail/model.h"

#include "api/diagnostics.h"
#include "io/ftp/descriptor.h"
#include "io/formats.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"

#include <cstdint>
#include <iomanip>
#include <ostream>
#include <string>
#include <utility>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {
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
                const auto code = model.op_codes()[node];

                switch (code) {
                case ir::pc::op_code::basic_event: {
                    const auto source = model.probability_source_refs()[node];
                    if (!source.valid() || source.raw() >= model.probability_sources().size()) {
                        diagnostics.push(api::make_error("io.ftp.unsupported-probability",
                                                         "FTP writer requires a probability source for every basic event"));
                        break;
                    }
                    const auto kind = model.probability_sources()[source.raw()].kind;
                    if (kind == ir::pc::probability_source_kind::unsupported) {
                        diagnostics.push(api::make_error("io.ftp.unsupported-probability",
                                                         "FTP writer cannot serialize unsupported probability sources"));
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
                output << ' ';
                if (edge.complement) {
                    output << '-';
                }
                output << nodeName(model, edge.source);
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
            auto const &probability = model.probability_sources()[source.raw()];
            if (probability.kind != ir::pc::probability_source_kind::literal) {
                return;
            }
            output << std::scientific << std::setprecision(6) << probability.literal << ' ' << nodeName(model, node) << '\n';
        }

    } // namespace

    auto write_pc(ir::pc::circuit const &model, std::ostream &output, io::write_request const &request) -> io::write_result {
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

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
