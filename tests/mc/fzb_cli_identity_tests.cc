#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef SCRAM_CLI_PATH
#error "SCRAM_CLI_PATH must be provided by CMake"
#endif
#ifndef FZB_PATH
#error "FZB_PATH must be provided by CMake"
#endif
#ifndef PROJECT_SOURCE_DIR
#error "PROJECT_SOURCE_DIR must be provided by CMake"
#endif

namespace {

struct HitsN {
  std::uint64_t hits = 0;
  std::uint64_t n = 0;
};

int run_cmd(const std::string &cmd) {
#ifdef _WIN32
  // MSVC std::system is cmd.exe /C; a command that starts with a quote is
  // parsed as the executable name. /S /C "..." takes the whole string.
  return std::system(("cmd /S /C \"" + cmd + "\"").c_str());
#else
  return std::system(cmd.c_str());
#endif
}

std::string shell_quote(const std::string &s) {
#ifdef _WIN32
  return "\"" + s + "\"";
#else
  return "'" + s + "'";
#endif
}

// RFC 4180 fields on a single line: device strings in the row are quoted.
std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> fields(1);
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quoted) {
      if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
        fields.back().push_back('"');
        ++i;
      } else if (c == '"') {
        quoted = false;
      } else {
        fields.back().push_back(c);
      }
    } else if (c == '"') {
      quoted = true;
    } else if (c == ',') {
      fields.emplace_back();
    } else {
      fields.back().push_back(c);
    }
  }
  return fields;
}

HitsN parse_convergence_csv(const std::filesystem::path &csv) {
  std::ifstream in(csv);
  REQUIRE(in);
  std::string header;
  REQUIRE(std::getline(in, header));
  std::string row;
  REQUIRE(std::getline(in, row));

  const auto keys = split_csv_line(header);
  const auto vals = split_csv_line(row);
  REQUIRE(keys.size() == vals.size());
  HitsN out;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (keys[i] == "one_bits") {
      out.hits = static_cast<std::uint64_t>(std::stoull(vals[i]));
    } else if (keys[i] == "total_bits") {
      out.n = static_cast<std::uint64_t>(std::stoull(vals[i]));
    }
  }
  REQUIRE(out.n > 0);
  return out;
}

HitsN parse_fzb_json(const std::string &json) {
  HitsN out;
  const auto grab = [&](const char *key) -> std::uint64_t {
    const std::string pat = std::string("\"") + key + "\": ";
    const auto pos = json.find(pat);
    REQUIRE(pos != std::string::npos);
    return static_cast<std::uint64_t>(std::stoull(json.substr(pos + pat.size())));
  };
  out.n = grab("n");
  out.hits = grab("hits");
  return out;
}

void check_n(const std::filesystem::path &xml, std::uint64_t requested) {
  REQUIRE(std::filesystem::exists(xml));

  const auto tmp = std::filesystem::temp_directory_path() /
                   ("fuzzybits-cli-id-" + xml.stem().string() + "-" + std::to_string(requested));
  // scram-cli appends to convergence.csv; a failed earlier run leaves its rows behind.
  std::filesystem::remove_all(tmp);
  std::filesystem::create_directories(tmp);
  const auto pcfb = tmp / (xml.stem().string() + ".pcfb");
  const auto report = tmp / "report.xml";
  const auto json_path = tmp / "fzb.json";
  const auto csv = tmp / "convergence.csv";

  std::ostringstream compile;
  compile << shell_quote(FZB_PATH) << " compile " << shell_quote(xml.string())
          << " -c 2 --no-kn -o " << shell_quote(pcfb.string());
  REQUIRE(run_cmd(compile.str()) == 0);
  REQUIRE(std::filesystem::exists(pcfb));

  std::ostringstream scram;
#ifdef _WIN32
  scram << "cmd /C \"cd /D " << shell_quote(tmp.string()) << " && " << shell_quote(SCRAM_CLI_PATH)
        << " --monte-carlo --probability --pdag -c 2 --no-kn -P wald -N " << requested
        << " --no-indent -V 0 -o " << shell_quote(report.string()) << " "
        << shell_quote(xml.string()) << "\"";
#else
  scram << "timeout 60 bash -lc \"cd '" << tmp.string() << "' && '" << SCRAM_CLI_PATH
        << "' --monte-carlo --probability --pdag -c 2 --no-kn -P wald -N " << requested
        << " --no-indent -V 0 -o '" << report.string() << "' '" << xml.string() << "'\"";
#endif
  REQUIRE(run_cmd(scram.str()) == 0);
  REQUIRE(std::filesystem::exists(csv));
  const auto from_scram = parse_convergence_csv(csv);

  std::ostringstream fzb;
#ifdef _WIN32
  fzb << shell_quote(FZB_PATH) << " run " << shell_quote(pcfb.string()) << " --trials " << requested
      << " --json > " << shell_quote(json_path.string());
#else
  fzb << "timeout 60 '" << FZB_PATH << "' run '" << pcfb.string() << "' --trials " << requested
      << " --json > '" << json_path.string() << "'";
#endif
  REQUIRE(run_cmd(fzb.str()) == 0);
  std::ifstream js(json_path);
  REQUIRE(js);
  std::ostringstream body;
  body << js.rdbuf();
  const auto from_fzb = parse_fzb_json(body.str());

  INFO("requested=" << requested << " scram hits=" << from_scram.hits << " n=" << from_scram.n
                    << " fzb hits=" << from_fzb.hits << " n=" << from_fzb.n);
  REQUIRE(from_fzb.n == from_scram.n);
  REQUIRE(from_fzb.hits == from_scram.hits);
  REQUIRE(from_fzb.n >= requested);

  std::filesystem::remove_all(tmp);
}

}  // namespace

TEST_CASE("fzb run matches scram-cli hits on chinese at 65536", "[mc][bit-identity][cli][n65536]") {
  check_n(std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "Aralia" / "chinese.xml", 65536);
}

TEST_CASE("fzb run matches scram-cli hits on chinese at 2^24", "[mc][bit-identity][cli][n2p24]") {
  check_n(std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "Aralia" / "chinese.xml", 1ull << 24);
}

TEST_CASE("fzb run matches scram-cli hits on SmallTree at 65536", "[mc][bit-identity][cli][smalltree]") {
  check_n(std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "SmallTree" / "SmallTree.xml", 65536);
}

TEST_CASE("fzb run matches scram-cli hits on BSCU at 65536", "[mc][bit-identity][cli][bscu]") {
  check_n(std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "BSCU" / "BSCU.xml", 65536);
}

TEST_CASE("fzb run rejects a non-default --seed", "[mc][cli][seed]") {
  const auto src = std::filesystem::path(PROJECT_SOURCE_DIR);
  const auto xml = src / "input" / "Aralia" / "chinese.xml";
  const auto tmp = std::filesystem::temp_directory_path() / "fuzzybits-seed-reject.pcfb";
  std::ostringstream compile;
  compile << shell_quote(FZB_PATH) << " compile " << shell_quote(xml.string())
          << " -c 2 --no-kn -o " << shell_quote(tmp.string());
  REQUIRE(run_cmd(compile.str()) == 0);
  std::ostringstream run;
#ifdef _WIN32
  run << shell_quote(FZB_PATH) << " run " << shell_quote(tmp.string())
      << " --trials 64 --seed 1 >NUL 2>&1";
#else
  run << "'" << FZB_PATH << "' run '" << tmp.string() << "' --trials 64 --seed 1 >/dev/null 2>&1";
#endif
  REQUIRE(run_cmd(run.str()) != 0);
  std::filesystem::remove(tmp);
}
