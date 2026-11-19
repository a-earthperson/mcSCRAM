#include "compile/exact_eval.h"
#include "compile/nombrilo.h"
#include "compile/pcfb_writer.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace {

using fuzzybits::compile::NombriloCircuit;
using fuzzybits::compile::NombriloCompileOptions;
using fuzzybits::compile::NombriloExportOptions;

std::uint32_t lit(std::uint32_t var, bool pos = true) { return (var << 1) | (pos ? 1u : 0u); }

NombriloCircuit make_circuit(std::uint32_t B, std::vector<std::uint32_t> ops,
                             std::vector<std::vector<std::uint32_t>> kids,
                             std::vector<std::pair<std::string, std::uint32_t>> names = {},
                             std::optional<std::vector<double>> weights = std::nullopt) {
  NombriloCircuit c;
  c.base_count = B;
  c.operators = std::move(ops);
  c.offsets.push_back(0);
  for (const auto &k : kids) {
    c.children.insert(c.children.end(), k.begin(), k.end());
    c.offsets.push_back(static_cast<std::uint32_t>(c.children.size()));
  }
  for (auto &[n, l] : names) {
    c.names.push_back(std::move(n));
    c.name_lits.push_back(l);
  }
  c.weights = std::move(weights);
  c.metadata["format"] = fuzzybits::compile::kNombriloFormat;
  c.metadata["version"] = fuzzybits::compile::kNombriloVersion;
  c.metadata["base_count"] = std::to_string(B);
  return c;
}

std::vector<std::byte> must_write(const NombriloCircuit &c) {
  auto bytes = fuzzybits::compile::write_nombrilo(c);
  REQUIRE(bytes);
  return std::move(*bytes);
}

NombriloCircuit must_read(std::span<const std::byte> bytes) {
  auto c = fuzzybits::compile::read_nombrilo(bytes);
  REQUIRE(c);
  return std::move(*c);
}

fuzzybits::compile::Compiled must_to_pcfb(const NombriloCircuit &c, NombriloCompileOptions opt = {}) {
  std::array<std::byte, 32> sha{};
  auto compiled = fuzzybits::compile::nombrilo_to_pcfb(c, opt, sha, "fixture");
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  const auto diags = pcfb::validate(*circuit);
  for (const auto &d : diags) {
    UNSCOPED_INFO(d.message);
  }
  REQUIRE(diags.empty());
  return std::move(*compiled);
}

bool arrays_equal(const NombriloCircuit &a, const NombriloCircuit &b) {
  if (a.base_count != b.base_count || a.operators != b.operators || a.offsets != b.offsets ||
      a.children != b.children || a.names != b.names || a.name_lits != b.name_lits) {
    return false;
  }
  if (a.weights.has_value() != b.weights.has_value()) {
    return false;
  }
  if (!a.weights) {
    return true;
  }
  if (a.weights->size() != b.weights->size()) {
    return false;
  }
  return std::memcmp(a.weights->data(), b.weights->data(), a.weights->size() * sizeof(double)) == 0;
}

std::filesystem::path nombrilo_dir() {
  if (const char *env = std::getenv("FUZZYBITS_NOMBRILO_DIR"); env && env[0]) {
    return env;
  }
  return std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "aralia-nombrilo";
}

}  // namespace

TEST_CASE("nombrilo round-trip write/read preserves arrays", "[nombrilo]") {
  auto c = make_circuit(2, {0, 1}, {{lit(1), lit(2)}, {lit(3), lit(1, false)}},
                        {{"a", lit(1)}, {"b", lit(2)}, {"g0", lit(3)}, {"r1", lit(4)}},
                        std::vector<double>{0.1, 0.2});
  auto bytes = must_write(c);
  auto back = must_read(bytes);
  REQUIRE(arrays_equal(c, back));
}

TEST_CASE("nombrilo AND/OR/XOR/ATLEAST and negation compile to valid pcfb", "[nombrilo]") {
  auto c = make_circuit(3,
                        {0, 1, 0xFFFFFFFFu, 2},
                        {{lit(1), lit(2)}, {lit(3), lit(1, false)}, {lit(3), lit(4)}, {lit(1), lit(2), lit(3)}},
                        {{"x", lit(1)}, {"y", lit(2)}, {"z", lit(3)}, {"r1", lit(7)}},
                        std::vector<double>{0.5, 0.25, 0.125});
  auto compiled = must_to_pcfb(c);
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  REQUIRE(fuzzybits::compile::random_sample_count(*circuit) == 3);
}

TEST_CASE("empty AND becomes ONES, empty OR/XOR become ZEROS, k>n becomes ZEROS", "[nombrilo][edge]") {
  auto c = make_circuit(1, {0, 1, 0xFFFFFFFFu, 5}, {{}, {}, {}, {lit(1)}},
                        {{"x", lit(1)}, {"r1", lit(5)}}, std::vector<double>{0.3});
  auto compiled = must_to_pcfb(c);
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  const auto *ops = circuit->subcircuits()->Get(0)->operators();
  REQUIRE(ops->size() >= 5);
  REQUIRE(ops->Get(2)->opcode() == pcfb::OpCode::BITWISE_ONES);    // empty AND, tensor 2
  REQUIRE(ops->Get(3)->opcode() == pcfb::OpCode::BITWISE_ZEROS);   // empty OR
  REQUIRE(ops->Get(4)->opcode() == pcfb::OpCode::BITWISE_ZEROS);   // empty XOR
  REQUIRE(ops->Get(5)->opcode() == pcfb::OpCode::BITWISE_ZEROS);   // atleast-5 of 1
  auto back = fuzzybits::compile::pcfb_to_nombrilo(*circuit, NombriloExportOptions{false});
  REQUIRE(back);
  REQUIRE(back->operators == c.operators);
  REQUIRE(back->children == c.children);
}

TEST_CASE("constant children and complementary literals are valid pcfb", "[nombrilo][edge]") {
  auto c = make_circuit(1, {0, 1}, {{lit(0), lit(1)}, {lit(1), lit(1, false)}},
                        {{"x", lit(1)}, {"r1", lit(3)}}, std::vector<double>{0.4});
  auto compiled = must_to_pcfb(c);
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
}

TEST_CASE("ATLEAST duplicate children: exact_eval counts multiplicity", "[nombrilo][edge]") {
  // ATLEAST-2 of [x, x] is x.
  auto c = make_circuit(1, {2}, {{lit(1), lit(1)}}, {{"x", lit(1)}, {"r1", lit(2)}},
                        std::vector<double>{0.3});
  auto compiled = must_to_pcfb(c);
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  const double p = fuzzybits::compile::exact_probability(*circuit);
  REQUIRE(p == Catch::Approx(0.3).margin(1e-15));
}

TEST_CASE("target on a negated literal appends NOT", "[nombrilo]") {
  auto c = make_circuit(1, {1}, {{lit(1)}}, {{"x", lit(1)}, {"nx", lit(1, false)}},
                        std::vector<double>{0.25});
  NombriloCompileOptions opt;
  opt.target = "nx";
  auto compiled = must_to_pcfb(c, opt);
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  bool saw_not = false;
  for (const auto *op : *circuit->subcircuits()->Get(0)->operators()) {
    if (op && op->opcode() == pcfb::OpCode::BITWISE_NOT) {
      saw_not = true;
    }
  }
  REQUIRE(saw_not);
  const double p = fuzzybits::compile::exact_probability(*circuit);
  REQUIRE(p == Catch::Approx(0.75).margin(1e-15));
}

TEST_CASE("names on negated literals and multiple names per variable survive round-trip", "[nombrilo]") {
  auto c = make_circuit(1, {0}, {{lit(1)}},
                        {{"x", lit(1)}, {"also_x", lit(1)}, {"not_x", lit(1, false)}, {"g", lit(2)}},
                        std::vector<double>{0.5});
  auto compiled = must_to_pcfb(c);
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  auto back = fuzzybits::compile::pcfb_to_nombrilo(*circuit, NombriloExportOptions{false});
  REQUIRE(back);
  REQUIRE(back->names == c.names);
  REQUIRE(back->name_lits == c.name_lits);
}

TEST_CASE("variant without weights inherits from explicit file", "[nombrilo]") {
  auto base = make_circuit(1, {0}, {{lit(1)}}, {{"x", lit(1)}, {"r1", lit(2)}},
                           std::vector<double>{0.01});
  auto var = base;
  var.weights.reset();
  auto base_bytes = must_write(base);
  auto var_bytes = must_write(var);
  const auto dir = std::filesystem::temp_directory_path();
  const auto base_path = dir / "nombrilo_base_fixture.safetensors";
  const auto var_path = dir / "nombrilo_base_fixture.e1.safetensors";
  {
    std::ofstream o(base_path, std::ios::binary);
    o.write(reinterpret_cast<const char *>(base_bytes.data()), static_cast<std::streamsize>(base_bytes.size()));
    std::ofstream v(var_path, std::ios::binary);
    v.write(reinterpret_cast<const char *>(var_bytes.data()), static_cast<std::streamsize>(var_bytes.size()));
  }
  auto compiled = fuzzybits::compile::compile_nombrilo_file(var_path.string(), {});
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
  auto back = fuzzybits::compile::pcfb_to_nombrilo(*circuit, NombriloExportOptions{false});
  REQUIRE(back);
  REQUIRE_FALSE(back->weights.has_value());
  std::filesystem::remove(base_path);
  std::filesystem::remove(var_path);
}

TEST_CASE("malformed nombrilo files are rejected", "[nombrilo][error]") {
  auto good = make_circuit(1, {0}, {{lit(1)}}, {{"x", lit(1)}, {"r1", lit(2)}},
                           std::vector<double>{0.5});

  SECTION("wrong format") {
    auto bytes = must_write(good);
    const auto needle = std::string("nombrilo.circuit");
    auto text = std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    auto pos = text.find(needle);
    REQUIRE(pos != std::string::npos);
    text.replace(pos, needle.size(), "nombrilo.circuiX");
    std::memcpy(bytes.data(), text.data(), bytes.size());
    auto c = fuzzybits::compile::read_nombrilo(bytes);
    REQUIRE_FALSE(c);
    REQUIRE(c.error().message.find("unsupported format") != std::string::npos);
  }
  SECTION("wrong version") {
    auto bytes = must_write(good);
    auto text = std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    auto pos = text.find("\"version\":\"1\"");
    REQUIRE(pos != std::string::npos);
    text.replace(pos, 13, "\"version\":\"9\"");
    std::memcpy(bytes.data(), text.data(), bytes.size());
    auto c = fuzzybits::compile::read_nombrilo(bytes);
    REQUIRE_FALSE(c);
    REQUIRE(c.error().message.find("version") != std::string::npos);
  }
  SECTION("weights length") {
    auto bad = good;
    bad.weights = std::vector<double>{0.1, 0.2};
    auto bytes = fuzzybits::compile::write_nombrilo(bad);
    REQUIRE_FALSE(bytes);
  }
  SECTION("p outside [0, 1]") {
    auto bad = good;
    (*bad.weights)[0] = 1.5;
    auto bytes = fuzzybits::compile::write_nombrilo(bad);
    REQUIRE_FALSE(bytes);
    REQUIRE(bytes.error().message.find("[0, 1]") != std::string::npos);
  }
  SECTION("forward reference") {
    auto bad = make_circuit(1, {0, 0}, {{lit(3)}, {lit(1)}}, {{"x", lit(1)}, {"r1", lit(3)}},
                            std::vector<double>{0.5});
    auto bytes = fuzzybits::compile::write_nombrilo(bad);
    REQUIRE_FALSE(bytes);
    REQUIRE(bytes.error().message.find("forward") != std::string::npos);
  }
  SECTION("literal out of range") {
    auto bad = make_circuit(1, {0}, {{lit(9)}}, {{"x", lit(1)}}, std::vector<double>{0.5});
    auto bytes = fuzzybits::compile::write_nombrilo(bad);
    REQUIRE_FALSE(bytes);
  }
  SECTION("names / name_lits mismatch") {
    auto bad = good;
    bad.name_lits.pop_back();
    auto bytes = fuzzybits::compile::write_nombrilo(bad);
    REQUIRE_FALSE(bytes);
  }
  SECTION("inconsistent offsets") {
    auto bad = good;
    bad.offsets.back() = 99;
    auto bytes = fuzzybits::compile::write_nombrilo(bad);
    REQUIRE_FALSE(bytes);
  }
}

TEST_CASE("pcfb NAND/NOR/XNOR/NOT/NO_OP/ATMOST/EXACTLY lower into nombrilo", "[nombrilo][pcfb]") {
  pcfb::builder b;
  auto a = b.add_sample(0.5, "a", 2);
  auto cnst = b.add_sample(0.25, "c", 3);
  REQUIRE(a);
  REQUIRE(cnst);
  const std::array<std::uint32_t, 2> ins{*a, *cnst};
  auto nand = b.add_op(pcfb::OpCode::BITWISE_NAND, ins);
  auto nor = b.add_op(pcfb::OpCode::BITWISE_NOR, ins);
  auto xnor = b.add_op(pcfb::OpCode::BITWISE_XNOR, ins);
  auto atleast = b.add_op(pcfb::OpCode::BITWISE_ATLEAST, ins, {}, 1);
  auto atmost = b.add_op(pcfb::OpCode::BITWISE_ATMOST, ins, {}, 0);
  auto exact = b.add_op(pcfb::OpCode::BITWISE_EXACTLY, ins, {}, 1);
  auto noop = b.add_op(pcfb::OpCode::BITWISE_NO_OP, std::array<std::uint32_t, 1>{*a});
  auto nota = b.add_op(pcfb::OpCode::BITWISE_NOT, std::array<std::uint32_t, 1>{*a});
  REQUIRE(nand);
  REQUIRE(nor);
  REQUIRE(xnor);
  REQUIRE(atleast);
  REQUIRE(atmost);
  REQUIRE(exact);
  REQUIRE(noop);
  REQUIRE(nota);
  REQUIRE(b.add_query(*exact, pcfb::QueryKind::PROBABILITY, "top"));
  auto bytes = b.finish();
  REQUIRE(bytes);
  auto circuit = pcfb::io::from_bytes(*bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());

  auto nombrilo = fuzzybits::compile::pcfb_to_nombrilo(*circuit, NombriloExportOptions{true});
  REQUIRE(nombrilo);
  REQUIRE(nombrilo->base_count == 2);
  REQUIRE_FALSE(nombrilo->operators.empty());
  REQUIRE(nombrilo->metadata.contains("fuzzybits.pcfb.queries"));
  REQUIRE(nombrilo->metadata.contains("fuzzybits.pcfb.event_ids"));

  auto strict = fuzzybits::compile::pcfb_to_nombrilo(*circuit, NombriloExportOptions{false});
  REQUIRE(strict);
  REQUIRE_FALSE(strict->metadata.contains("fuzzybits.pcfb.queries"));

  std::array<std::byte, 32> sha{};
  auto restored = fuzzybits::compile::nombrilo_to_pcfb(*nombrilo, {}, sha, "from-pcfb");
  REQUIRE(restored);
  auto restored_c = pcfb::io::from_bytes(restored->bytes);
  REQUIRE(restored_c);
  REQUIRE(pcfb::validate(*restored_c).empty());
  const double p0 = fuzzybits::compile::exact_probability(*circuit);
  const double p1 = fuzzybits::compile::exact_probability(*restored_c);
  REQUIRE(p1 == Catch::Approx(p0).margin(1e-12));
}

TEST_CASE("nombrilo metadata records source format and no compile.* keys", "[nombrilo]") {
  auto c = make_circuit(1, {0}, {{lit(1)}}, {{"x", lit(1)}, {"r1", lit(2)}},
                        std::vector<double>{0.01});
  auto compiled = must_to_pcfb(c);
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  bool saw_source = false;
  bool saw_compile = false;
  for (const auto *kv : *circuit->get()->metadata()) {
    if (!kv || !kv->key()) {
      continue;
    }
    const auto key = kv->key()->str();
    if (key == "general.source_format") {
      REQUIRE(kv->value_as_StringValue());
      REQUIRE(kv->value_as_StringValue()->value()->str() == fuzzybits::compile::kNombriloSourceFormat);
      saw_source = true;
    }
    if (key.rfind("compile.", 0) == 0) {
      saw_compile = true;
    }
  }
  REQUIRE(saw_source);
  REQUIRE_FALSE(saw_compile);
}

TEST_CASE("dataset nombrilo → pcfb → nombrilo is array-exact", "[nombrilo][dataset]") {
  const auto dir = nombrilo_dir();
  if (!std::filesystem::is_directory(dir)) {
    SKIP("FUZZYBITS_NOMBRILO_DIR not set and " + dir.string() + " is absent");
  }
  std::size_t n = 0;
  std::size_t exact = 0;
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".safetensors") {
      continue;
    }
    ++n;
    auto orig = fuzzybits::compile::read_nombrilo_file(entry.path().string());
    REQUIRE(orig);
    const bool had_weights = orig->weights.has_value();
    NombriloCompileOptions opt;
    auto compiled = fuzzybits::compile::compile_nombrilo_file(entry.path().string(), opt);
    REQUIRE(compiled);
    auto circuit = pcfb::io::from_bytes(compiled->bytes);
    REQUIRE(circuit);
    REQUIRE(pcfb::validate(*circuit).empty());
    auto back = fuzzybits::compile::pcfb_to_nombrilo(*circuit, NombriloExportOptions{false});
    REQUIRE(back);
    REQUIRE(back->operators == orig->operators);
    REQUIRE(back->offsets == orig->offsets);
    REQUIRE(back->children == orig->children);
    REQUIRE(back->names == orig->names);
    REQUIRE(back->name_lits == orig->name_lits);
    REQUIRE(back->metadata["format"] == "nombrilo.circuit");
    REQUIRE(back->metadata["version"] == "1");
    REQUIRE(back->metadata["base_count"] == orig->metadata["base_count"]);
    if (had_weights) {
      REQUIRE(back->weights);
      REQUIRE(back->weights->size() == orig->weights->size());
      REQUIRE(std::memcmp(back->weights->data(), orig->weights->data(),
                          orig->weights->size() * sizeof(double)) == 0);
    } else {
      REQUIRE_FALSE(back->weights.has_value());
    }
    ++exact;
  }
  REQUIRE(n == 311);
  REQUIRE(exact == 311);
}

void collect_sample_ids(const pcfb::ir::Circuit &circuit, std::vector<std::uint32_t> &ids,
                        std::vector<double> &ps) {
  REQUIRE(circuit.subcircuits());
  REQUIRE(circuit.subcircuits()->size() > 0);
  const auto *ops = circuit.subcircuits()->Get(0)->operators();
  REQUIRE(ops);
  for (const auto *op : *ops) {
    if (!op || op->opcode() != pcfb::OpCode::SAMPLE) {
      continue;
    }
    const auto *sa = op->args_as_SampleArgs();
    REQUIRE(sa);
    ids.push_back(sa->event_id());
    ps.push_back(sa->probability());
  }
}

void mef_extras_roundtrip(const std::filesystem::path &xml) {
  auto compiled = fuzzybits::compile::compile_files({xml.string()}, {});
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());

  auto extras = fuzzybits::compile::pcfb_to_nombrilo(*circuit, NombriloExportOptions{true});
  REQUIRE(extras);
  REQUIRE(extras->metadata.contains("fuzzybits.pcfb.event_ids"));
  REQUIRE(extras->metadata.contains("fuzzybits.pcfb.queries"));
  REQUIRE(extras->metadata.contains("fuzzybits.pcfb.sources"));
  REQUIRE(extras->metadata.contains("fuzzybits.pcfb.metadata"));

  std::array<std::byte, 32> sha{};
  auto restored = fuzzybits::compile::nombrilo_to_pcfb(*extras, {}, sha, xml.stem().string());
  REQUIRE(restored);
  auto rc = pcfb::io::from_bytes(restored->bytes);
  REQUIRE(rc);
  REQUIRE(pcfb::validate(*rc).empty());

  std::vector<std::uint32_t> ids0, ids1;
  std::vector<double> p0s, p1s;
  collect_sample_ids(*circuit, ids0, p0s);
  collect_sample_ids(*rc, ids1, p1s);
  REQUIRE(ids0 == ids1);
  REQUIRE(p0s.size() == p1s.size());
  for (std::size_t i = 0; i < p0s.size(); ++i) {
    REQUIRE(p0s[i] == p1s[i]);
  }
  REQUIRE(circuit->subcircuits()->Get(0)->queries()->size() == rc->subcircuits()->Get(0)->queries()->size());

  const auto nrand = fuzzybits::compile::random_sample_count(*circuit);
  if (nrand <= 20) {
    const double p0 = fuzzybits::compile::exact_probability(*circuit);
    const double p1 = fuzzybits::compile::exact_probability(*rc);
    REQUIRE(p1 == Catch::Approx(p0).margin(1e-12));
  }

  auto strict = fuzzybits::compile::pcfb_to_nombrilo(*circuit, NombriloExportOptions{false});
  REQUIRE(strict);
  REQUIRE_FALSE(strict->metadata.contains("fuzzybits.pcfb.queries"));
  REQUIRE_FALSE(strict->metadata.contains("fuzzybits.pcfb.event_ids"));
  auto restored_strict = fuzzybits::compile::nombrilo_to_pcfb(*strict, {}, sha, xml.stem().string());
  REQUIRE(restored_strict);
  auto rs = pcfb::io::from_bytes(restored_strict->bytes);
  REQUIRE(rs);
  REQUIRE(pcfb::validate(*rs).empty());
  if (nrand <= 20) {
    const double p0 = fuzzybits::compile::exact_probability(*circuit);
    const double p1 = fuzzybits::compile::exact_probability(*rs);
    REQUIRE(p1 == Catch::Approx(p0).margin(1e-12));
  }
}

TEST_CASE("MEF pcfb → nombrilo → pcfb extras restore SAMPLE identity", "[nombrilo][mef]") {
  const auto xml = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "SmallTree" / "SmallTree.xml";
  if (!std::filesystem::is_regular_file(xml)) {
    SKIP("SmallTree.xml not in the source tree");
  }
  mef_extras_roundtrip(xml);
}

TEST_CASE("Aralia MEF extras round-trip restores SAMPLE identity", "[nombrilo][mef][dataset]") {
  const auto dir = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "Aralia";
  if (!std::filesystem::is_directory(dir)) {
    SKIP("input/Aralia is absent");
  }
  std::size_t n = 0;
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".xml") {
      continue;
    }
    INFO(entry.path().filename().string());
    mef_extras_roundtrip(entry.path());
    ++n;
  }
  REQUIRE(n == 43);
}

TEST_CASE("chinese nombrilo exact probability matches MEF compile", "[nombrilo][dataset]") {
  const auto xml = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "Aralia" / "chinese.xml";
  const auto nom = nombrilo_dir() / "chinese.safetensors";
  if (!std::filesystem::is_regular_file(xml) || !std::filesystem::is_regular_file(nom)) {
    SKIP("chinese.xml or chinese.safetensors absent");
  }
  auto mef = fuzzybits::compile::compile_files({xml.string()}, {});
  REQUIRE(mef);
  auto mef_c = pcfb::io::from_bytes(mef->bytes);
  REQUIRE(mef_c);
  auto nom_c = fuzzybits::compile::compile_nombrilo_file(nom.string(), {});
  REQUIRE(nom_c);
  auto nom_ir = pcfb::io::from_bytes(nom_c->bytes);
  REQUIRE(nom_ir);
  const double p_mef = fuzzybits::compile::exact_probability(*mef_c);
  const double p_nom = fuzzybits::compile::exact_probability(*nom_ir);
  REQUIRE(p_nom == Catch::Approx(p_mef).epsilon(1e-8));
}
