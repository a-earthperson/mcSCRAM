#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "api/diagnostics.h"
#include "io/ftp/descriptor.h"
#include "io/formats.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

    [[nodiscard]] auto diagnosticsText(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
        std::ostringstream output;
        for (auto const &diagnostic : diagnostics.items()) {
            output << diagnostic.code << ": " << diagnostic.message << '\n';
        }
        return output.str();
    }

    [[nodiscard]] auto checksum(std::string_view text) -> std::uint64_t {
        auto value = std::uint64_t{14695981039346656037ULL};
        for (const auto character : text) {
            value ^= static_cast<unsigned char>(character);
            value *= std::uint64_t{1099511628211ULL};
        }
        return value;
    }

    [[nodiscard]] auto hasFtpExtension(std::filesystem::path const &path) -> bool {
        auto extension = path.extension().string();
        std::ranges::transform(extension, extension.begin(), [](const unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        return extension == ".ftp";
    }

    [[nodiscard]] auto inputRoot() -> std::filesystem::path {
        return std::filesystem::path(PROJECT_SOURCE_DIR) / "input";
    }

    void collectFtpFiles(std::filesystem::path const &root, std::vector<std::filesystem::path> &files) {
        std::error_code error;
        auto iterator = std::filesystem::directory_iterator(root, std::filesystem::directory_options::skip_permission_denied, error);
        if (error) {
            return;
        }

        for (const auto end = std::filesystem::directory_iterator{}; iterator != end; iterator.increment(error)) {
            if (error) {
                error.clear();
                continue;
            }

            auto const &entry = *iterator;
            if (entry.is_directory(error)) {
                if (!error) {
                    collectFtpFiles(entry.path(), files);
                }
                error.clear();
                continue;
            }
            if (entry.is_regular_file(error) && hasFtpExtension(entry.path())) {
                files.push_back(entry.path());
            }
            error.clear();
        }
    }

    [[nodiscard]] auto ftpCorpusFiles() -> std::vector<std::filesystem::path> {
        std::vector<std::filesystem::path> files;
        collectFtpFiles(inputRoot(), files);
        std::ranges::sort(files, [](auto const &lhs, auto const &rhs) {
            return lhs.generic_string() < rhs.generic_string();
        });
        return files;
    }

    [[nodiscard]] auto readFile(std::filesystem::path const &path) -> fuzzybits::io::backend::fuzzybits::ftp::ftp_read_result {
        std::ifstream input(path);
        REQUIRE(input);
        return fuzzybits::io::backend::fuzzybits::ftp::read(input);
    }

    [[nodiscard]] auto requirePcModel(fuzzybits::io::backend::fuzzybits::ftp::read_output const &read)
        -> fuzzybits::ir::pc::circuit const & {
        auto const *pc = fuzzybits::ir::pc::circuit_from(read.model);
        REQUIRE(pc != nullptr);
        REQUIRE(pc->valid());
        REQUIRE(pc->node_count() > 0);
        REQUIRE_FALSE(pc->targets().empty());
        return *pc;
    }

    [[nodiscard]] auto writeOutput(fuzzybits::ir::pc::circuit const &model, std::optional<double> truncationLimit = std::nullopt)
        -> std::string {
        fuzzybits::io::write_request request{
            .format = fuzzybits::io::formats::ftp(),
        };
        if (truncationLimit) {
            request.options.set(fuzzybits::io::backend::fuzzybits::ftp::truncation_limit_option(), *truncationLimit);
        }

        std::ostringstream output;
        const auto written = fuzzybits::io::backend::fuzzybits::ftp::write(model.to_handle(), output, request);
        INFO(diagnosticsText(written.diagnostics()));
        REQUIRE(written);
        return output.str();
    }

} // namespace

TEST_CASE("FTP writer descriptor advertises the FTP format", "[io][ftp]") {
    const auto descriptor = fuzzybits::io::backend::fuzzybits::ftp::writer_descriptor();

    REQUIRE(descriptor.provider == fuzzybits::api::provider_id("fuzzybits"));
    REQUIRE(descriptor.endpoint == fuzzybits::io::backend::fuzzybits::ftp::writer_endpoint_id());
    REQUIRE(descriptor.output_format == fuzzybits::io::formats::ftp());
}

TEST_CASE("FTP writer round-trips the writable PC subset", "[io][ftp]") {
    fuzzybits::ir::pc::builder builder;
    const auto aProbability = builder.add_literal_probability(0.1);
    const auto bProbability = builder.add_literal_probability(0.2);
    const auto a = builder.add_basic_event(aProbability);
    const auto b = builder.add_basic_event(bProbability);
    const auto top = builder.add_and({
        fuzzybits::ir::pc::edge_ref{.source = a},
        fuzzybits::ir::pc::edge_ref{.source = b, .complement = true},
    });
    const auto target = builder.add_target(top);
    (void)target;

    auto finalized = std::move(builder).seal();
    REQUIRE(finalized.ok());

    const auto text = writeOutput(finalized.output, 1.0e-6);

    REQUIRE(text.find("LIMIT") != std::string::npos);
    REQUIRE(text.find("T0 + N") != std::string::npos);
    REQUIRE(text.find(" -E") != std::string::npos);

    std::istringstream input(text);
    auto read = fuzzybits::io::backend::fuzzybits::ftp::read(input);
    INFO(diagnosticsText(read.diagnostics()));
    REQUIRE(read);

    auto const *pc = fuzzybits::ir::pc::circuit_from(read.value().model);
    REQUIRE(pc != nullptr);
    REQUIRE(pc->targets().size() == 1);
    REQUIRE(pc->probability_sources().size() == 2);
    const auto operands = pc->operands(pc->targets()[0].node);
    REQUIRE(operands.size() == 2);
    REQUIRE(operands[1].complement);
}

TEST_CASE("FTP input corpus round-trips to stable write checksums", "[io][ftp][corpus]") {
    const auto files = ftpCorpusFiles();
    REQUIRE_FALSE(files.empty());

    for (auto const &file : files) {
        DYNAMIC_SECTION(file.lexically_relative(inputRoot()).generic_string()) {
            auto firstRead = readFile(file);
            INFO(diagnosticsText(firstRead.diagnostics()));
            REQUIRE(firstRead);

            auto const &firstPc = requirePcModel(firstRead.value());
            const auto firstOutput = writeOutput(firstPc, firstRead.value().metadata.truncation_limit);
            std::istringstream reparsedInput(firstOutput);
            auto secondRead = fuzzybits::io::backend::fuzzybits::ftp::read(reparsedInput);
            INFO(diagnosticsText(secondRead.diagnostics()));
            REQUIRE(secondRead);

            auto const &secondPc = requirePcModel(secondRead.value());
            const auto secondOutput = writeOutput(secondPc, secondRead.value().metadata.truncation_limit);
            INFO("first checksum: " << checksum(firstOutput));
            INFO("second checksum: " << checksum(secondOutput));
            REQUIRE(firstOutput.size() == secondOutput.size());
            REQUIRE(checksum(firstOutput) == checksum(secondOutput));
        }
    }
}
