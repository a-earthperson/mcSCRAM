#include "compile/exact_eval.h"
#include "compile/pcfb_writer.h"
#include "compile/sha256.h"
#include "version.h"

#ifdef FZB_HAS_RUNTIME
#include "runtime/runtime.h"
#endif

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace po = boost::program_options;

namespace {

void print_root_usage(std::ostream &out) {
  out << "Usage: fzb <command> [options]\n\n"
      << "Commands:\n"
      << "  compile   OpenPSA/MEF XML → preprocessed PDAG → .pcfb\n"
      << "  inspect   Summarise, validate, or exactly evaluate a .pcfb\n"
      << "  run       Monte Carlo a .pcfb (estimate and CI per query)\n";
}

std::string json_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          std::ostringstream os;
          os << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
          out += os.str();
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

#ifdef FZB_HAS_RUNTIME
std::uint64_t parse_trials(const std::string &text) {
  char *end = nullptr;
  const double v = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || v <= 0 || !std::isfinite(v)) {
    throw fuzzybits::runtime::RuntimeError("invalid --trials '" + text + "'");
  }
  return static_cast<std::uint64_t>(v);
}

int cmd_run(int argc, char **argv) {
  po::options_description desc("fzb run options");
  desc.add_options()
      ("help,h", "display this help message")
      ("trials,N", po::value<std::string>(), "Bernoulli trials (e.g. 1e9, 1000000)")
      ("seed,S", po::value<int>()->default_value(372),
       "legacy Settings::seed (Bernoulli Philox is counter-based and ignores this)")
      ("backend", po::value<std::string>(), "AdaptiveCpp backend: cuda or omp (sets ACPP_VISIBILITY_MASK first)")
      ("json", "print results as JSON")
      ("input-file", po::value<std::string>(), ".pcfb file");

  po::positional_options_description pos;
  pos.add("input-file", 1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << "Option error: " << ex.what() << "\n\n" << desc << std::endl;
    return 1;
  }
  if (vm.count("help")) {
    std::cout << "Usage: fzb run model.pcfb --trials N [--seed S] [--backend cuda|omp] [--json]\n\n"
              << desc << std::endl;
    return 0;
  }
  if (!vm.count("input-file")) {
    std::cerr << "fzb run: missing .pcfb file\n\n" << desc << std::endl;
    return 1;
  }
  if (!vm.count("trials")) {
    std::cerr << "fzb run: --trials is required\n\n" << desc << std::endl;
    return 1;
  }

  fuzzybits::runtime::RunOptions opt;
  try {
    opt.trials = parse_trials(vm["trials"].as<std::string>());
  } catch (const std::exception &ex) {
    std::cerr << "fzb run: " << ex.what() << std::endl;
    return 1;
  }
  opt.seed = vm["seed"].as<int>();
  if (vm.count("backend")) {
    opt.backend = vm["backend"].as<std::string>();
  }

  // ACPP_VISIBILITY_MASK must be set before AdaptiveCpp constructs a queue.
  try {
    fuzzybits::runtime::apply_backend_mask(opt.backend);
  } catch (const std::exception &ex) {
    std::cerr << "fzb run: " << ex.what() << std::endl;
    return 1;
  }

  try {
    const auto results = fuzzybits::runtime::run_pcfb(vm["input-file"].as<std::string>(), opt);
    if (vm.count("json")) {
      std::cout << "{\n  \"queries\": [\n";
      for (std::size_t i = 0; i < results.size(); ++i) {
        const auto &r = results[i];
        std::cout << "    {\"name\": \"" << json_escape(r.name) << "\", "
                  << "\"estimate\": " << std::setprecision(16) << std::scientific << r.estimate << ", "
                  << "\"ci_low\": " << r.ci_low << ", "
                  << "\"ci_high\": " << r.ci_high << ", "
                  << "\"n\": " << std::dec << r.n << ", "
                  << "\"hits\": " << r.hits << "}";
        if (i + 1 < results.size()) {
          std::cout << ",";
        }
        std::cout << "\n";
      }
      std::cout << "  ]\n}\n";
    } else {
      for (const auto &r : results) {
        std::cout << r.name << "  " << std::setprecision(8) << std::scientific << r.estimate << "  ["
                  << r.ci_low << ", " << r.ci_high << "]  n=" << std::dec << r.n << "  hits=" << r.hits
                  << "\n";
      }
    }
    return 0;
  } catch (const fuzzybits::runtime::InvalidPcfbError &ex) {
    std::cerr << "fzb run: " << ex.what() << std::endl;
    for (const auto &d : ex.diagnostics) {
      std::cerr << "validate: " << d.location << ": " << d.message << std::endl;
    }
    return 1;
  } catch (const std::exception &ex) {
    std::cerr << "fzb run: " << ex.what() << std::endl;
    return 1;
  }
}
#else
int cmd_run(int, char **) {
  std::cerr << "fzb run requires FUZZYBITS_WITH_MC=ON (the SYCL Monte Carlo engine).\n";
  return 2;
}
#endif

const char *opcode_name(pcfb::OpCode op) {
  switch (op) {
    case pcfb::OpCode::BITWISE_NO_OP:
      return "NO_OP";
    case pcfb::OpCode::BITWISE_NOT:
      return "NOT";
    case pcfb::OpCode::BITWISE_AND:
      return "AND";
    case pcfb::OpCode::BITWISE_OR:
      return "OR";
    case pcfb::OpCode::BITWISE_XOR:
      return "XOR";
    case pcfb::OpCode::BITWISE_NAND:
      return "NAND";
    case pcfb::OpCode::BITWISE_NOR:
      return "NOR";
    case pcfb::OpCode::BITWISE_XNOR:
      return "XNOR";
    case pcfb::OpCode::BITWISE_ATLEAST:
      return "ATLEAST";
    case pcfb::OpCode::BITWISE_ATMOST:
      return "ATMOST";
    case pcfb::OpCode::BITWISE_EXACTLY:
      return "EXACTLY";
    case pcfb::OpCode::BITWISE_ZEROS:
      return "ZEROS";
    case pcfb::OpCode::BITWISE_ONES:
      return "ONES";
    case pcfb::OpCode::SAMPLE:
      return "SAMPLE";
    default:
      return "RESERVED";
  }
}

const char *source_name(pcfb::SampleSource s) {
  switch (s) {
    case pcfb::SampleSource::BASIC_EVENT:
      return "BASIC_EVENT";
    case pcfb::SampleSource::COLLAPSED_MODULE:
      return "COLLAPSED_MODULE";
    case pcfb::SampleSource::HOUSE_EVENT:
      return "HOUSE_EVENT";
    default:
      return "UNKNOWN";
  }
}

struct InspectSummary {
  std::uint32_t schema_version = 0;
  std::size_t n_tensors = 0;
  std::size_t n_ops = 0;
  std::size_t n_queries = 0;
  std::size_t n_samples = 0;
  std::map<std::string, int> opcode_counts;
  std::map<std::string, int> sample_sources;
  std::map<std::string, std::string> metadata;
  std::vector<std::string> queries;
};

InspectSummary summarise(const pcfb::ir::Circuit &circuit) {
  InspectSummary s;
  s.schema_version = circuit.schema_version();
  if (circuit.get() && circuit.get()->metadata()) {
    for (const auto *kv : *circuit.get()->metadata()) {
      if (!kv || !kv->key()) {
        continue;
      }
      std::string value;
      if (const auto *str = kv->value_as_StringValue(); str && str->value()) {
        value = str->value()->str();
      } else if (const auto *i = kv->value_as_IntValue()) {
        value = std::to_string(i->value());
      } else if (const auto *f = kv->value_as_FloatValue()) {
        std::ostringstream os;
        os << f->value();
        value = os.str();
      } else if (const auto *b = kv->value_as_BytesValue(); b && b->value()) {
        std::vector<std::uint8_t> raw(b->value()->begin(), b->value()->end());
        if (raw.size() == 32) {
          std::array<std::uint8_t, 32> digest{};
          std::copy(raw.begin(), raw.end(), digest.begin());
          value = fuzzybits::compile::sha256_hex(digest);
        } else {
          value = "<" + std::to_string(raw.size()) + " bytes>";
        }
      }
      s.metadata.emplace(kv->key()->str(), std::move(value));
    }
  }
  if (!circuit.subcircuits() || circuit.subcircuits()->size() == 0) {
    return s;
  }
  const auto *sc = circuit.subcircuits()->Get(0);
  if (sc->tensors()) {
    s.n_tensors = sc->tensors()->size();
  }
  if (sc->operators()) {
    s.n_ops = sc->operators()->size();
    for (const auto *op : *sc->operators()) {
      if (!op) {
        continue;
      }
      ++s.opcode_counts[opcode_name(op->opcode())];
      if (op->opcode() == pcfb::OpCode::SAMPLE) {
        ++s.n_samples;
        if (const auto *sa = op->args_as_SampleArgs()) {
          ++s.sample_sources[source_name(sa->source())];
        }
      }
    }
  }
  if (sc->queries()) {
    s.n_queries = sc->queries()->size();
    for (const auto *q : *sc->queries()) {
      if (!q) {
        continue;
      }
      std::string name = q->name() ? q->name()->str() : "";
      s.queries.push_back(name.empty() ? ("t" + std::to_string(q->target())) : name);
    }
  }
  return s;
}

void print_text(const InspectSummary &s, std::ostream &out) {
  out << "schema_version  0x" << std::hex << s.schema_version << std::dec << "\n";
  out << "tensors         " << s.n_tensors << "\n";
  out << "operators       " << s.n_ops << "\n";
  out << "samples         " << s.n_samples << "\n";
  out << "queries         " << s.n_queries << "\n";
  if (!s.opcode_counts.empty()) {
    out << "opcodes\n";
    for (const auto &[name, n] : s.opcode_counts) {
      out << "  " << name << "  " << n << "\n";
    }
  }
  if (!s.sample_sources.empty()) {
    out << "sample_source\n";
    for (const auto &[name, n] : s.sample_sources) {
      out << "  " << name << "  " << n << "\n";
    }
  }
  if (!s.queries.empty()) {
    out << "query_names\n";
    for (const auto &q : s.queries) {
      out << "  " << q << "\n";
    }
  }
  if (!s.metadata.empty()) {
    out << "metadata\n";
    for (const auto &[k, v] : s.metadata) {
      out << "  " << k << "  " << v << "\n";
    }
  }
}

void print_json(const InspectSummary &s, std::ostream &out) {
  out << "{\n";
  out << "  \"schema_version\": " << s.schema_version << ",\n";
  out << "  \"tensors\": " << s.n_tensors << ",\n";
  out << "  \"operators\": " << s.n_ops << ",\n";
  out << "  \"samples\": " << s.n_samples << ",\n";
  out << "  \"queries\": " << s.n_queries << ",\n";
  out << "  \"opcodes\": {";
  bool first = true;
  for (const auto &[name, n] : s.opcode_counts) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << "\"" << json_escape(name) << "\": " << n;
  }
  out << "},\n  \"sample_source\": {";
  first = true;
  for (const auto &[name, n] : s.sample_sources) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << "\"" << json_escape(name) << "\": " << n;
  }
  out << "},\n  \"query_names\": [";
  first = true;
  for (const auto &q : s.queries) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << "\"" << json_escape(q) << "\"";
  }
  out << "],\n  \"metadata\": {";
  first = true;
  for (const auto &[k, v] : s.metadata) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << "\"" << json_escape(k) << "\": \"" << json_escape(v) << "\"";
  }
  out << "}\n}\n";
}

int cmd_compile(int argc, char **argv) {
  po::options_description desc("fzb compile options");
  desc.add_options()
      ("help,h", "display this help message")
      ("output,o", po::value<std::string>(), "output .pcfb path")
      ("compilation-passes,c", po::value<int>()->default_value(2),
       "0=off 1=null-only 2=optimize 4-8=multipass")
      ("no-kn", "expand k/n to and/or")
      ("no-xor", "expand xor to and/or")
      ("recognize-atleast", "recognize expanded threshold groups (default on)")
      ("no-recognize-atleast", "disable k/n group recognition")
      ("recognize-xor", "recognize expanded xor (default on)")
      ("no-recognize-xor", "disable xor group recognition")
      ("keep-null", "keep NULL/pass-through gates")
      ("ccf", "expand common-cause failure groups")
      ("allow-extern", "**UNSAFE** allow external libraries")
      ("target", po::value<std::string>(), "top-event / gate id")
      ("input-file", po::value<std::vector<std::string>>(), "MEF input files");

  po::positional_options_description pos;
  pos.add("input-file", -1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << "Option error: " << ex.what() << "\n\n" << desc << std::endl;
    return 1;
  }
  if (vm.count("help")) {
    std::cout << "Usage: fzb compile model.xml [-c N] [--no-kn] [-o model.pcfb]\n\n" << desc << std::endl;
    return 0;
  }
  if (!vm.count("input-file")) {
    std::cerr << "fzb compile: missing input file\n\n" << desc << std::endl;
    return 1;
  }

  fuzzybits::compile::Options opt;
  opt.compilation_level = vm["compilation-passes"].as<int>();
  opt.expand_atleast_gates = vm.count("no-kn") > 0;
  opt.expand_xor_gates = vm.count("no-xor") > 0;
  if (vm.count("no-recognize-atleast")) {
    opt.recognize_atleast_groups = false;
  } else if (vm.count("recognize-atleast")) {
    opt.recognize_atleast_groups = true;
  }
  if (vm.count("no-recognize-xor")) {
    opt.recognize_xor_groups = false;
  } else if (vm.count("recognize-xor")) {
    opt.recognize_xor_groups = true;
  }
  opt.keep_null_gates = vm.count("keep-null") > 0;
  opt.ccf = vm.count("ccf") > 0;
  opt.allow_extern = vm.count("allow-extern") > 0;
  if (vm.count("target")) {
    opt.target = vm["target"].as<std::string>();
  }

  const auto files = vm["input-file"].as<std::vector<std::string>>();
  auto compiled = fuzzybits::compile::compile_files(files, opt);
  if (!compiled) {
    std::cerr << "fzb compile: " << compiled.error().message << std::endl;
    return 1;
  }

  std::filesystem::path out_path;
  if (vm.count("output")) {
    out_path = vm["output"].as<std::string>();
  } else {
    out_path = std::filesystem::path(files.front()).stem();
    out_path += ".pcfb";
  }
  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::cerr << "fzb compile: cannot write " << out_path << std::endl;
    return 1;
  }
  out.write(reinterpret_cast<const char *>(compiled->bytes.data()),
            static_cast<std::streamsize>(compiled->bytes.size()));
  if (!out) {
    std::cerr << "fzb compile: write failed: " << out_path << std::endl;
    return 1;
  }
  std::cerr << "wrote " << out_path << " (" << compiled->bytes.size() << " bytes)"
            << " model=" << compiled->model_name << " top=" << compiled->target_id << std::endl;
  return 0;
}

int cmd_inspect(int argc, char **argv) {
  po::options_description desc("fzb inspect options");
  desc.add_options()
      ("help,h", "display this help message")
      ("validate", "run pcfb::validate; exit non-zero on diagnostics")
      ("json", "print summary as JSON")
      ("exact", "exhaustive exact P(query) for small models (≤ 30 events)")
      ("input-file", po::value<std::string>(), ".pcfb file");

  po::positional_options_description pos;
  pos.add("input-file", 1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << "Option error: " << ex.what() << "\n\n" << desc << std::endl;
    return 1;
  }
  if (vm.count("help")) {
    std::cout << "Usage: fzb inspect model.pcfb [--validate] [--json] [--exact]\n\n" << desc << std::endl;
    return 0;
  }
  if (!vm.count("input-file")) {
    std::cerr << "fzb inspect: missing .pcfb file\n\n" << desc << std::endl;
    return 1;
  }

  const auto path = vm["input-file"].as<std::string>();
  auto circuit = pcfb::io::read(path);
  if (!circuit) {
    std::cerr << "fzb inspect: " << circuit.error().message << std::endl;
    return 1;
  }

  int rc = 0;
  if (vm.count("validate")) {
    const auto diags = pcfb::validate(*circuit);
    if (!diags.empty()) {
      rc = 1;
      for (const auto &d : diags) {
        std::cerr << "validate: " << d.location << ": " << d.message << std::endl;
      }
    } else if (!vm.count("json")) {
      std::cerr << "validate: ok\n";
    }
  }

  const auto summary = summarise(*circuit);
  if (vm.count("json")) {
    print_json(summary, std::cout);
  } else {
    print_text(summary, std::cout);
  }

  if (vm.count("exact")) {
    try {
      const double p = fuzzybits::compile::exact_probability(*circuit);
      std::cout << std::setprecision(16) << std::scientific << "exact_probability  " << p << std::endl;
    } catch (const fuzzybits::compile::ExactEvalError &ex) {
      std::cerr << "fzb inspect --exact: " << ex.what() << std::endl;
      return 1;
    }
  }
  return rc;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    print_root_usage(std::cerr);
    return 1;
  }
  const std::string cmd = argv[1];
  if (cmd == "-h" || cmd == "--help" || cmd == "help") {
    print_root_usage(std::cout);
    return 0;
  }
  if (cmd == "-v" || cmd == "--version" || cmd == "version") {
    std::cout << "fzb " << SCRAM_VERSION << " (" << SCRAM_GIT_REVISION << ")\n";
    return 0;
  }
  if (cmd == "compile") {
    return cmd_compile(argc - 1, argv + 1);
  }
  if (cmd == "inspect") {
    return cmd_inspect(argc - 1, argv + 1);
  }
  if (cmd == "run") {
    return cmd_run(argc - 1, argv + 1);
  }
  std::cerr << "fzb: unknown command '" << cmd << "'\n";
  print_root_usage(std::cerr);
  return 1;
}
