#include "compile/nombrilo.h"

#include "compile/sha256.h"
#include "version.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace fuzzybits::compile {
namespace {

pcfb::Error nombrilo_error(std::string message) {
  return pcfb::Error{pcfb::Error::Code::InvalidBuffer, std::move(message)};
}

pcfb::Error io_error(std::string message) {
  return pcfb::Error{pcfb::Error::Code::IoError, std::move(message)};
}

std::string generator_version() {
  std::string v = SCRAM_VERSION;
  if (constexpr const char *rev = SCRAM_GIT_REVISION; rev && rev[0] != '\0' && std::string_view(rev) != "unknown") {
    v += "+g";
    v += rev;
  }
  return v;
}

constexpr std::uint32_t kXorCode = 0xFFFFFFFFu;
constexpr std::size_t kMaxHeaderBytes = 16u * 1024u * 1024u;

std::uint32_t lit_var(std::uint32_t lit) { return lit >> 1; }
bool lit_pos(std::uint32_t lit) { return (lit & 1u) != 0; }
std::uint32_t make_lit(std::uint32_t var, bool positive) { return (var << 1) | (positive ? 1u : 0u); }

std::uint32_t read_u32_le(const std::byte *p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t read_u64_le(const std::byte *p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
  }
  return v;
}

void write_u32_le(std::vector<std::byte> &out, std::uint32_t v) {
  out.push_back(static_cast<std::byte>(v & 0xffu));
  out.push_back(static_cast<std::byte>((v >> 8) & 0xffu));
  out.push_back(static_cast<std::byte>((v >> 16) & 0xffu));
  out.push_back(static_cast<std::byte>((v >> 24) & 0xffu));
}

void write_u64_le(std::vector<std::byte> &out, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xffu));
  }
}

void write_f64_le(std::vector<std::byte> &out, double v) {
  std::uint64_t bits = 0;
  static_assert(sizeof(double) == 8);
  std::memcpy(&bits, &v, 8);
  write_u64_le(out, bits);
}

double read_f64_le(const std::byte *p) {
  const std::uint64_t bits = read_u64_le(p);
  double v = 0;
  std::memcpy(&v, &bits, 8);
  return v;
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
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

std::string json_string(std::string_view s) {
  std::string out = "\"";
  out += json_escape(s);
  out += '"';
  return out;
}

std::string json_u32_array(std::span<const std::uint32_t> xs) {
  std::string out = "[";
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (i) {
      out += ',';
    }
    out += std::to_string(xs[i]);
  }
  out += ']';
  return out;
}

std::string hex_encode(std::span<const std::byte> bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.resize(bytes.size() * 2);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const auto v = static_cast<unsigned>(bytes[i]);
    out[2 * i] = kHex[v >> 4];
    out[2 * i + 1] = kHex[v & 0xf];
  }
  return out;
}

pcfb::Result<std::vector<std::byte>> hex_decode(std::string_view hex) {
  if (hex.size() % 2 != 0) {
    return std::unexpected(nombrilo_error("odd-length hex string"));
  }
  auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
      return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
      return c - 'A' + 10;
    }
    return -1;
  };
  std::vector<std::byte> out(hex.size() / 2);
  for (std::size_t i = 0; i < out.size(); ++i) {
    const int hi = nibble(hex[2 * i]);
    const int lo = nibble(hex[2 * i + 1]);
    if (hi < 0 || lo < 0) {
      return std::unexpected(nombrilo_error("invalid hex digit"));
    }
    out[i] = static_cast<std::byte>((hi << 4) | lo);
  }
  return out;
}

struct JsonCursor {
  std::string_view s;
  std::size_t i = 0;

  void skip() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
      ++i;
    }
  }
  bool eof() const { return i >= s.size(); }
  char peek() {
    skip();
    return eof() ? '\0' : s[i];
  }
  bool eat(char c) {
    skip();
    if (eof() || s[i] != c) {
      return false;
    }
    ++i;
    return true;
  }
};

pcfb::Result<void> expect(JsonCursor &c, char ch, const char *what) {
  if (!c.eat(ch)) {
    return std::unexpected(nombrilo_error(std::string("JSON: expected ") + what));
  }
  return {};
}

pcfb::Result<std::string> parse_json_string(JsonCursor &c) {
  c.skip();
  if (c.eof() || c.s[c.i] != '"') {
    return std::unexpected(nombrilo_error("JSON: expected string"));
  }
  ++c.i;
  std::string out;
  while (c.i < c.s.size()) {
    const unsigned char ch = static_cast<unsigned char>(c.s[c.i]);
    ++c.i;
    if (ch == '"') {
      return out;
    }
    if (ch == '\\') {
      if (c.i >= c.s.size()) {
        return std::unexpected(nombrilo_error("JSON: truncated escape"));
      }
      const char e = c.s[c.i++];
      switch (e) {
        case '"':
        case '\\':
        case '/':
          out.push_back(e);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          if (c.i + 4 > c.s.size()) {
            return std::unexpected(nombrilo_error("JSON: truncated \\u escape"));
          }
          unsigned code = 0;
          for (int k = 0; k < 4; ++k) {
            const char h = c.s[c.i++];
            code <<= 4;
            if (h >= '0' && h <= '9') {
              code |= static_cast<unsigned>(h - '0');
            } else if (h >= 'a' && h <= 'f') {
              code |= static_cast<unsigned>(h - 'a' + 10);
            } else if (h >= 'A' && h <= 'F') {
              code |= static_cast<unsigned>(h - 'A' + 10);
            } else {
              return std::unexpected(nombrilo_error("JSON: invalid \\u escape"));
            }
          }
          if (code < 0x80) {
            out.push_back(static_cast<char>(code));
          } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xc0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
          } else {
            out.push_back(static_cast<char>(0xe0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
          }
          break;
        }
        default:
          return std::unexpected(nombrilo_error("JSON: unknown escape"));
      }
      continue;
    }
    if (ch < 0x20) {
      return std::unexpected(nombrilo_error("JSON: unescaped control character"));
    }
    out.push_back(static_cast<char>(ch));
  }
  return std::unexpected(nombrilo_error("JSON: unterminated string"));
}

pcfb::Result<std::uint64_t> parse_json_u64(JsonCursor &c) {
  c.skip();
  if (c.eof() || !std::isdigit(static_cast<unsigned char>(c.s[c.i]))) {
    return std::unexpected(nombrilo_error("JSON: expected unsigned integer"));
  }
  std::uint64_t v = 0;
  while (c.i < c.s.size() && std::isdigit(static_cast<unsigned char>(c.s[c.i]))) {
    const std::uint64_t d = static_cast<unsigned>(c.s[c.i] - '0');
    if (v > (std::numeric_limits<std::uint64_t>::max() - d) / 10) {
      return std::unexpected(nombrilo_error("JSON: integer overflow"));
    }
    v = v * 10 + d;
    ++c.i;
  }
  return v;
}

pcfb::Result<void> skip_json_value(JsonCursor &c);

pcfb::Result<void> skip_json_array(JsonCursor &c) {
  if (auto r = expect(c, '[', "'['"); !r) {
    return r;
  }
  if (c.peek() == ']') {
    c.eat(']');
    return {};
  }
  for (;;) {
    if (auto r = skip_json_value(c); !r) {
      return r;
    }
    if (c.eat(']')) {
      return {};
    }
    if (auto r = expect(c, ',', "',' or ']'"); !r) {
      return r;
    }
  }
}

pcfb::Result<void> skip_json_object(JsonCursor &c) {
  if (auto r = expect(c, '{', "'{'"); !r) {
    return r;
  }
  if (c.peek() == '}') {
    c.eat('}');
    return {};
  }
  for (;;) {
    if (auto key = parse_json_string(c); !key) {
      return std::unexpected(key.error());
    }
    if (auto r = expect(c, ':', "':'"); !r) {
      return r;
    }
    if (auto r = skip_json_value(c); !r) {
      return r;
    }
    if (c.eat('}')) {
      return {};
    }
    if (auto r = expect(c, ',', "',' or '}'"); !r) {
      return r;
    }
  }
}

pcfb::Result<void> skip_json_value(JsonCursor &c) {
  const char p = c.peek();
  if (p == '"') {
    auto s = parse_json_string(c);
    if (!s) {
      return std::unexpected(s.error());
    }
    return {};
  }
  if (p == '{') {
    return skip_json_object(c);
  }
  if (p == '[') {
    return skip_json_array(c);
  }
  if (p == 't') {
    if (c.i + 4 <= c.s.size() && c.s.substr(c.i, 4) == "true") {
      c.i += 4;
      return {};
    }
  }
  if (p == 'f') {
    if (c.i + 5 <= c.s.size() && c.s.substr(c.i, 5) == "false") {
      c.i += 5;
      return {};
    }
  }
  if (p == 'n') {
    if (c.i + 4 <= c.s.size() && c.s.substr(c.i, 4) == "null") {
      c.i += 4;
      return {};
    }
  }
  if (p == '-' || std::isdigit(static_cast<unsigned char>(p))) {
    if (p == '-') {
      ++c.i;
    }
    if (auto n = parse_json_u64(c); !n) {
      return std::unexpected(n.error());
    }
    if (c.i < c.s.size() && c.s[c.i] == '.') {
      ++c.i;
      if (auto n = parse_json_u64(c); !n) {
        return std::unexpected(n.error());
      }
    }
    if (c.i < c.s.size() && (c.s[c.i] == 'e' || c.s[c.i] == 'E')) {
      ++c.i;
      if (c.i < c.s.size() && (c.s[c.i] == '+' || c.s[c.i] == '-')) {
        ++c.i;
      }
      if (auto n = parse_json_u64(c); !n) {
        return std::unexpected(n.error());
      }
    }
    return {};
  }
  return std::unexpected(nombrilo_error("JSON: unexpected value"));
}

struct TensorInfo {
  std::string dtype;
  std::uint64_t numel = 0;
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};

struct SafetensorsHeader {
  std::map<std::string, std::string> metadata;
  std::map<std::string, TensorInfo> tensors;
};

pcfb::Result<std::vector<std::uint64_t>> parse_json_u64_array(JsonCursor &c) {
  if (auto r = expect(c, '[', "'['"); !r) {
    return std::unexpected(r.error());
  }
  std::vector<std::uint64_t> out;
  if (c.peek() == ']') {
    c.eat(']');
    return out;
  }
  for (;;) {
    auto n = parse_json_u64(c);
    if (!n) {
      return std::unexpected(n.error());
    }
    out.push_back(*n);
    if (c.eat(']')) {
      return out;
    }
    if (auto r = expect(c, ',', "',' or ']'"); !r) {
      return std::unexpected(r.error());
    }
  }
}

pcfb::Result<TensorInfo> parse_tensor_info(JsonCursor &c) {
  if (auto r = expect(c, '{', "tensor object"); !r) {
    return std::unexpected(r.error());
  }
  TensorInfo info;
  bool have_dtype = false;
  bool have_shape = false;
  bool have_off = false;
  if (c.peek() == '}') {
    return std::unexpected(nombrilo_error("empty tensor info"));
  }
  for (;;) {
    auto key = parse_json_string(c);
    if (!key) {
      return std::unexpected(key.error());
    }
    if (auto r = expect(c, ':', "':'"); !r) {
      return std::unexpected(r.error());
    }
    if (*key == "dtype") {
      auto s = parse_json_string(c);
      if (!s) {
        return std::unexpected(s.error());
      }
      info.dtype = std::move(*s);
      have_dtype = true;
    } else if (*key == "shape") {
      auto shape = parse_json_u64_array(c);
      if (!shape) {
        return std::unexpected(shape.error());
      }
      if (shape->size() != 1) {
        return std::unexpected(nombrilo_error("nombrilo tensors must be rank-1"));
      }
      info.numel = (*shape)[0];
      have_shape = true;
    } else if (*key == "data_offsets") {
      auto off = parse_json_u64_array(c);
      if (!off) {
        return std::unexpected(off.error());
      }
      if (off->size() != 2) {
        return std::unexpected(nombrilo_error("data_offsets must be [start, end]"));
      }
      info.begin = (*off)[0];
      info.end = (*off)[1];
      have_off = true;
    } else {
      if (auto r = skip_json_value(c); !r) {
        return std::unexpected(r.error());
      }
    }
    if (c.eat('}')) {
      break;
    }
    if (auto r = expect(c, ',', "',' or '}'"); !r) {
      return std::unexpected(r.error());
    }
  }
  if (!have_dtype || !have_shape || !have_off) {
    return std::unexpected(nombrilo_error("tensor info missing dtype/shape/data_offsets"));
  }
  if (info.end < info.begin) {
    return std::unexpected(nombrilo_error("tensor data_offsets end < start"));
  }
  return info;
}

pcfb::Result<std::map<std::string, std::string>> parse_metadata_object(JsonCursor &c) {
  if (auto r = expect(c, '{', "metadata object"); !r) {
    return std::unexpected(r.error());
  }
  std::map<std::string, std::string> out;
  if (c.peek() == '}') {
    c.eat('}');
    return out;
  }
  for (;;) {
    auto key = parse_json_string(c);
    if (!key) {
      return std::unexpected(key.error());
    }
    if (auto r = expect(c, ':', "':'"); !r) {
      return std::unexpected(r.error());
    }
    auto val = parse_json_string(c);
    if (!val) {
      return std::unexpected(val.error());
    }
    out.emplace(std::move(*key), std::move(*val));
    if (c.eat('}')) {
      return out;
    }
    if (auto r = expect(c, ',', "',' or '}'"); !r) {
      return std::unexpected(r.error());
    }
  }
}

pcfb::Result<SafetensorsHeader> parse_safetensors_header(std::string_view json) {
  JsonCursor c{json, 0};
  if (auto r = expect(c, '{', "header object"); !r) {
    return std::unexpected(r.error());
  }
  SafetensorsHeader h;
  if (c.peek() == '}') {
    return h;
  }
  for (;;) {
    auto key = parse_json_string(c);
    if (!key) {
      return std::unexpected(key.error());
    }
    if (auto r = expect(c, ':', "':'"); !r) {
      return std::unexpected(r.error());
    }
    if (*key == "__metadata__") {
      auto meta = parse_metadata_object(c);
      if (!meta) {
        return std::unexpected(meta.error());
      }
      h.metadata = std::move(*meta);
    } else {
      auto info = parse_tensor_info(c);
      if (!info) {
        return std::unexpected(info.error());
      }
      h.tensors.emplace(std::move(*key), std::move(*info));
    }
    if (c.eat('}')) {
      break;
    }
    if (auto r = expect(c, ',', "',' or '}'"); !r) {
      return std::unexpected(r.error());
    }
  }
  return h;
}

pcfb::Result<std::vector<std::uint32_t>> parse_u32_array_text(std::string_view json) {
  JsonCursor c{json, 0};
  auto xs = parse_json_u64_array(c);
  if (!xs) {
    return std::unexpected(xs.error());
  }
  std::vector<std::uint32_t> out;
  out.reserve(xs->size());
  for (auto v : *xs) {
    if (v > std::numeric_limits<std::uint32_t>::max()) {
      return std::unexpected(nombrilo_error("u32 array element out of range"));
    }
    out.push_back(static_cast<std::uint32_t>(v));
  }
  return out;
}

void child_order_warn(std::string_view detail) {
  std::fprintf(stderr, "pcfb_to_nombrilo: ignoring nombrilo.child_order (%.*s); using canonical child order\n",
               static_cast<int>(detail.size()), detail.data());
}

bool children_are_canonical(std::span<const std::uint32_t> kids) {
  bool saw_neg = false;
  for (auto lit : kids) {
    if (!lit_pos(lit)) {
      saw_neg = true;
    } else if (saw_neg) {
      return false;
    }
  }
  return true;
}

std::string dump_child_order(const NombriloCircuit &c) {
  std::string out = "{";
  bool any = false;
  for (std::size_t i = 0; i < c.operators.size(); ++i) {
    const auto lo = c.offsets[i];
    const auto hi = c.offsets[i + 1];
    const auto kids = std::span<const std::uint32_t>(c.children.data() + lo, hi - lo);
    if (children_are_canonical(kids)) {
      continue;
    }
    std::vector<std::uint32_t> pol;
    pol.reserve(kids.size());
    for (auto lit : kids) {
      pol.push_back(lit_pos(lit) ? 1u : 0u);
    }
    if (any) {
      out += ',';
    }
    any = true;
    out += json_string(std::to_string(i));
    out += ':';
    out += json_u32_array(pol);
  }
  out += '}';
  return any ? out : std::string{};
}

pcfb::Result<std::map<std::uint32_t, std::vector<std::uint32_t>>> parse_child_order_object(std::string_view json) {
  JsonCursor c{json, 0};
  if (auto r = expect(c, '{', "child_order object"); !r) {
    return std::unexpected(r.error());
  }
  std::map<std::uint32_t, std::vector<std::uint32_t>> out;
  if (c.peek() == '}') {
    c.eat('}');
    return out;
  }
  for (;;) {
    auto key = parse_json_string(c);
    if (!key) {
      return std::unexpected(key.error());
    }
    if (auto r = expect(c, ':', "':'"); !r) {
      return std::unexpected(r.error());
    }
    auto pol = parse_json_u64_array(c);
    if (!pol) {
      return std::unexpected(pol.error());
    }
    char *end = nullptr;
    const auto idx = std::strtoul(key->c_str(), &end, 10);
    if (end == key->c_str() || *end != '\0' || idx > std::numeric_limits<std::uint32_t>::max()) {
      return std::unexpected(nombrilo_error("child_order key is not a gate index"));
    }
    std::vector<std::uint32_t> bits;
    bits.reserve(pol->size());
    for (auto v : *pol) {
      if (v > 1) {
        return std::unexpected(nombrilo_error("child_order polarity must be 0 or 1"));
      }
      bits.push_back(static_cast<std::uint32_t>(v));
    }
    out[static_cast<std::uint32_t>(idx)] = std::move(bits);
    if (c.eat('}')) {
      return out;
    }
    if (auto r = expect(c, ',', "',' or '}'"); !r) {
      return std::unexpected(r.error());
    }
  }
}

std::vector<std::uint32_t> apply_child_order(const std::vector<std::uint32_t> &pos,
                                            const std::vector<std::uint32_t> &neg,
                                            const std::vector<std::uint32_t> *order, std::uint32_t gate) {
  std::vector<std::uint32_t> canonical;
  canonical.reserve(pos.size() + neg.size());
  canonical.insert(canonical.end(), pos.begin(), pos.end());
  canonical.insert(canonical.end(), neg.begin(), neg.end());
  if (!order) {
    return canonical;
  }
  if (order->size() != canonical.size()) {
    child_order_warn("gate " + std::to_string(gate) + " length mismatch");
    return canonical;
  }
  std::vector<std::uint32_t> out;
  out.reserve(order->size());
  std::size_t ip = 0;
  std::size_t in = 0;
  for (auto bit : *order) {
    if (bit == 1) {
      if (ip >= pos.size()) {
        child_order_warn("gate " + std::to_string(gate) + " too many positive slots");
        return canonical;
      }
      out.push_back(pos[ip++]);
    } else if (bit == 0) {
      if (in >= neg.size()) {
        child_order_warn("gate " + std::to_string(gate) + " too many inverted slots");
        return canonical;
      }
      out.push_back(neg[in++]);
    } else {
      child_order_warn("gate " + std::to_string(gate) + " polarity not 0/1");
      return canonical;
    }
  }
  if (ip != pos.size() || in != neg.size()) {
    child_order_warn("gate " + std::to_string(gate) + " unused inputs");
    return canonical;
  }
  return out;
}

struct NamesBlob {
  std::vector<std::string> names;
  std::vector<std::uint32_t> lits;
};

pcfb::Result<NamesBlob> parse_names_blob(std::string_view json) {
  JsonCursor c{json, 0};
  if (auto r = expect(c, '{', "names object"); !r) {
    return std::unexpected(r.error());
  }
  NamesBlob out;
  bool have_names = false;
  bool have_lits = false;
  if (c.peek() == '}') {
    return out;
  }
  for (;;) {
    auto key = parse_json_string(c);
    if (!key) {
      return std::unexpected(key.error());
    }
    if (auto r = expect(c, ':', "':'"); !r) {
      return std::unexpected(r.error());
    }
    if (*key == "names") {
      if (auto r = expect(c, '[', "'['"); !r) {
        return std::unexpected(r.error());
      }
      if (c.peek() != ']') {
        for (;;) {
          auto s = parse_json_string(c);
          if (!s) {
            return std::unexpected(s.error());
          }
          out.names.push_back(std::move(*s));
          if (c.eat(']')) {
            break;
          }
          if (auto r = expect(c, ',', "',' or ']'"); !r) {
            return std::unexpected(r.error());
          }
        }
      } else {
        c.eat(']');
      }
      have_names = true;
    } else if (*key == "lits") {
      auto xs = parse_json_u64_array(c);
      if (!xs) {
        return std::unexpected(xs.error());
      }
      out.lits.reserve(xs->size());
      for (auto v : *xs) {
        if (v > std::numeric_limits<std::uint32_t>::max()) {
          return std::unexpected(nombrilo_error("name literal out of range"));
        }
        out.lits.push_back(static_cast<std::uint32_t>(v));
      }
      have_lits = true;
    } else {
      if (auto r = skip_json_value(c); !r) {
        return std::unexpected(r.error());
      }
    }
    if (c.eat('}')) {
      break;
    }
    if (auto r = expect(c, ',', "',' or '}'"); !r) {
      return std::unexpected(r.error());
    }
  }
  if (have_names != have_lits || out.names.size() != out.lits.size()) {
    return std::unexpected(nombrilo_error("nombrilo.names names/lits mismatch"));
  }
  return out;
}

std::string names_blob_json(const NombriloCircuit &c) {
  std::string out = "{\"names\":[";
  for (std::size_t i = 0; i < c.names.size(); ++i) {
    if (i) {
      out += ',';
    }
    out += json_string(c.names[i]);
  }
  out += "],\"lits\":";
  out += json_u32_array(c.name_lits);
  out += '}';
  return out;
}

pcfb::Result<std::vector<std::byte>> load_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return std::unexpected(io_error("cannot open " + path));
  }
  in.seekg(0, std::ios::end);
  const auto n = in.tellg();
  if (n < 0) {
    return std::unexpected(io_error("cannot stat " + path));
  }
  in.seekg(0, std::ios::beg);
  std::vector<std::byte> bytes(static_cast<std::size_t>(n));
  if (n > 0 && !in.read(reinterpret_cast<char *>(bytes.data()), n)) {
    return std::unexpected(io_error("cannot read " + path));
  }
  return bytes;
}

pcfb::Result<std::span<const std::byte>> tensor_bytes(std::span<const std::byte> data, const TensorInfo &info,
                                                      std::string_view name, std::size_t elem_size) {
  if (info.end > data.size() || info.begin > info.end) {
    return std::unexpected(nombrilo_error(std::string(name) + " data_offsets out of range"));
  }
  const auto n = info.end - info.begin;
  if (info.numel > (std::numeric_limits<std::uint64_t>::max() / elem_size)) {
    return std::unexpected(nombrilo_error(std::string(name) + " numel overflow"));
  }
  if (n != info.numel * elem_size) {
    return std::unexpected(nombrilo_error(std::string(name) + " byte length does not match shape"));
  }
  return data.subspan(static_cast<std::size_t>(info.begin), static_cast<std::size_t>(n));
}

bool dtype_is(std::string_view got, std::string_view a, std::string_view b) {
  return got == a || got == b;
}

pcfb::Result<std::vector<std::uint32_t>> load_u32(std::span<const std::byte> data, const TensorInfo &info,
                                                  std::string_view name) {
  if (!dtype_is(info.dtype, "U32", "u32")) {
    return std::unexpected(nombrilo_error(std::string(name) + " dtype must be U32"));
  }
  auto bytes = tensor_bytes(data, info, name, 4);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  std::vector<std::uint32_t> out(info.numel);
  for (std::uint64_t i = 0; i < info.numel; ++i) {
    out[static_cast<std::size_t>(i)] = read_u32_le(bytes->data() + i * 4);
  }
  return out;
}

pcfb::Result<std::vector<double>> load_f64(std::span<const std::byte> data, const TensorInfo &info,
                                           std::string_view name) {
  if (!dtype_is(info.dtype, "F64", "f64")) {
    return std::unexpected(nombrilo_error(std::string(name) + " dtype must be F64"));
  }
  auto bytes = tensor_bytes(data, info, name, 8);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  std::vector<double> out(info.numel);
  for (std::uint64_t i = 0; i < info.numel; ++i) {
    out[static_cast<std::size_t>(i)] = read_f64_le(bytes->data() + i * 8);
  }
  return out;
}

pcfb::Result<std::vector<std::string>> split_names(std::span<const std::byte> raw) {
  std::vector<std::string> names;
  std::string cur;
  for (auto b : raw) {
    if (b == std::byte{0}) {
      names.push_back(std::move(cur));
      cur.clear();
    } else {
      cur.push_back(static_cast<char>(b));
    }
  }
  if (!cur.empty()) {
    names.push_back(std::move(cur));
  }
  return names;
}

pcfb::Result<void> validate_circuit(const NombriloCircuit &c) {
  const auto B = c.base_count;
  const auto G = c.operators.size();
  if (c.offsets.size() != G + 1) {
    return std::unexpected(nombrilo_error("offsets length must be G+1"));
  }
  if (G > 0 && c.offsets[0] != 0) {
    return std::unexpected(nombrilo_error("offsets[0] must be 0"));
  }
  if (G == 0 && (c.offsets.empty() || c.offsets[0] != 0)) {
    return std::unexpected(nombrilo_error("offsets must be [0] when there are no gates"));
  }
  for (std::size_t i = 1; i < c.offsets.size(); ++i) {
    if (c.offsets[i] < c.offsets[i - 1]) {
      return std::unexpected(nombrilo_error("offsets must be non-decreasing"));
    }
  }
  if (c.offsets.empty() || c.offsets.back() != c.children.size()) {
    return std::unexpected(nombrilo_error("offsets[G] must equal children length"));
  }
  const auto n_vars = static_cast<std::uint64_t>(B) + 1 + G;  // 0..B+G
  if (n_vars > (std::numeric_limits<std::uint32_t>::max() >> 1)) {
    return std::unexpected(nombrilo_error("too many variables for 32-bit literals"));
  }
  for (std::size_t i = 0; i < G; ++i) {
    const auto lo = c.offsets[i];
    const auto hi = c.offsets[i + 1];
    const auto gate_var = B + 1 + static_cast<std::uint32_t>(i);
    for (auto j = lo; j < hi; ++j) {
      const auto lit = c.children[j];
      const auto v = lit_var(lit);
      if (v >= n_vars) {
        return std::unexpected(nombrilo_error("child literal " + std::to_string(lit) + " is out of range"));
      }
      if (v >= gate_var) {
        return std::unexpected(nombrilo_error("gate " + std::to_string(i) + " has a forward reference to variable " +
                                             std::to_string(v)));
      }
    }
  }
  if (c.names.size() != c.name_lits.size()) {
    return std::unexpected(nombrilo_error("names and name_lits length mismatch"));
  }
  std::unordered_set<std::string> seen;
  seen.reserve(c.names.size());
  for (std::size_t i = 0; i < c.names.size(); ++i) {
    if (c.names[i].empty()) {
      return std::unexpected(nombrilo_error("empty name is not allowed"));
    }
    if (!seen.insert(c.names[i]).second) {
      return std::unexpected(nombrilo_error("duplicate name '" + c.names[i] + "'"));
    }
    const auto lit = c.name_lits[i];
    if (lit_var(lit) >= n_vars) {
      return std::unexpected(nombrilo_error("name '" + c.names[i] + "' literal is out of range"));
    }
  }
  if (c.weights) {
    if (c.weights->size() != B) {
      return std::unexpected(nombrilo_error("weights length must equal base_count"));
    }
    for (std::size_t i = 0; i < c.weights->size(); ++i) {
      const double p = (*c.weights)[i];
      if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
        return std::unexpected(nombrilo_error("weights[" + std::to_string(i) + "] is outside [0, 1]"));
      }
    }
  }
  return {};
}

std::string meta_get(const NombriloCircuit &c, std::string_view key) {
  auto it = c.metadata.find(std::string(key));
  return it == c.metadata.end() ? std::string() : it->second;
}

std::vector<std::pair<std::string, std::uint32_t>> input_name_table(const NombriloCircuit &c) {
  std::vector<std::pair<std::string, std::uint32_t>> out;
  for (std::size_t i = 0; i < c.names.size(); ++i) {
    const auto v = lit_var(c.name_lits[i]);
    if (v >= 1 && v <= c.base_count) {
      out.emplace_back(c.names[i], c.name_lits[i]);
    }
  }
  return out;
}

std::string first_positive_name(const NombriloCircuit &c, std::uint32_t var) {
  std::string any;
  for (std::size_t i = 0; i < c.names.size(); ++i) {
    if (lit_var(c.name_lits[i]) != var) {
      continue;
    }
    if (any.empty()) {
      any = c.names[i];
    }
    if (lit_pos(c.name_lits[i])) {
      return c.names[i];
    }
  }
  return any;
}

std::optional<std::uint32_t> lookup_name(const NombriloCircuit &c, std::string_view name) {
  for (std::size_t i = 0; i < c.names.size(); ++i) {
    if (c.names[i] == name) {
      return c.name_lits[i];
    }
  }
  return std::nullopt;
}

bool is_variant_suffix(std::string_view s) {
  if (s.size() < 2) {
    return false;
  }
  const char kind = static_cast<char>(std::tolower(static_cast<unsigned char>(s[0])));
  if (kind != 'e' && kind != 'f') {
    return false;
  }
  for (std::size_t i = 1; i < s.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
      return false;
    }
  }
  return true;
}

std::optional<std::string> sibling_base_path(const std::string &source_path) {
  const std::filesystem::path p(source_path);
  const auto stem = p.stem().string();
  const auto dot = stem.rfind('.');
  if (dot == std::string::npos || dot == 0) {
    return std::nullopt;
  }
  if (!is_variant_suffix(std::string_view(stem).substr(dot + 1))) {
    return std::nullopt;
  }
  auto base = p.parent_path() / (stem.substr(0, dot) + p.extension().string());
  if (!std::filesystem::is_regular_file(base)) {
    return std::nullopt;
  }
  return base.string();
}

std::string tensor_json(std::string_view dtype, std::uint64_t numel, std::uint64_t begin, std::uint64_t end) {
  std::string out = "{\"dtype\":";
  out += json_string(dtype);
  out += ",\"shape\":[";
  out += std::to_string(numel);
  out += "],\"data_offsets\":[";
  out += std::to_string(begin);
  out += ',';
  out += std::to_string(end);
  out += "]}";
  return out;
}

pcfb::Result<void> set_meta(pcfb::builder &b, std::string_view key, std::string_view value) {
  return b.set(key, value);
}

pcfb::Result<void> write_nombrilo_side_meta(pcfb::builder &b, const NombriloCircuit &circuit, bool inherited) {
  if (auto r = set_meta(b, "nombrilo.names", names_blob_json(circuit)); !r) {
    return r;
  }
  if (auto order = dump_child_order(circuit); !order.empty()) {
    if (auto r = set_meta(b, "nombrilo.child_order", order); !r) {
      return r;
    }
  }
  if (auto r = set_meta(b, "nombrilo.weights_source", inherited ? "inherited" : "inline"); !r) {
    return r;
  }
  if (inherited) {
    if (auto file = meta_get(circuit, "nombrilo.weights_file"); !file.empty()) {
      if (auto r = set_meta(b, "nombrilo.weights_file", file); !r) {
        return r;
      }
    }
  }
  return {};
}

std::optional<std::string> pcfb_meta_string(const pcfb::ir::Circuit &circuit, std::string_view key) {
  if (!circuit.get() || !circuit.get()->metadata()) {
    return std::nullopt;
  }
  for (const auto *kv : *circuit.get()->metadata()) {
    if (!kv || !kv->key() || kv->key()->str() != key) {
      continue;
    }
    if (const auto *s = kv->value_as_StringValue()) {
      if (s->value()) {
        return s->value()->str();
      }
    }
    if (const auto *i = kv->value_as_IntValue()) {
      return std::to_string(i->value());
    }
    if (const auto *f = kv->value_as_FloatValue()) {
      std::ostringstream os;
      os << std::setprecision(17) << f->value();
      return os.str();
    }
    if (const auto *by = kv->value_as_BytesValue()) {
      if (by->value()) {
        const auto *v = by->value();
        return hex_encode(std::span<const std::byte>(reinterpret_cast<const std::byte *>(v->data()), v->size()));
      }
    }
  }
  return std::nullopt;
}

std::string dump_pcfb_metadata_json(const pcfb::ir::Circuit &circuit) {
  std::string out = "[";
  bool first = true;
  if (!circuit.get() || !circuit.get()->metadata()) {
    out += ']';
    return out;
  }
  for (const auto *kv : *circuit.get()->metadata()) {
    if (!kv || !kv->key()) {
      continue;
    }
    const std::string key = kv->key()->str();
    if (!first) {
      out += ',';
    }
    first = false;
    out += "{\"key\":";
    out += json_string(key);
    if (const auto *s = kv->value_as_StringValue()) {
      out += ",\"t\":\"s\",\"v\":";
      out += json_string(s->value() ? s->value()->str() : "");
    } else if (const auto *i = kv->value_as_IntValue()) {
      out += ",\"t\":\"i\",\"v\":\"";
      out += std::to_string(i->value());
      out += '"';
    } else if (const auto *f = kv->value_as_FloatValue()) {
      out += ",\"t\":\"f\",\"v\":\"";
      std::ostringstream os;
      os << std::setprecision(17) << f->value();
      out += os.str();
      out += '"';
    } else if (const auto *by = kv->value_as_BytesValue()) {
      out += ",\"t\":\"b\",\"v\":";
      if (by->value()) {
        const auto *v = by->value();
        out += json_string(
            hex_encode(std::span<const std::byte>(reinterpret_cast<const std::byte *>(v->data()), v->size())));
      } else {
        out += "\"\"";
      }
    } else {
      out += ",\"t\":\"s\",\"v\":\"\"";
    }
    out += '}';
  }
  out += ']';
  return out;
}

pcfb::Result<void> restore_pcfb_metadata(pcfb::builder &b, std::string_view json) {
  JsonCursor c{json, 0};
  if (auto r = expect(c, '[', "'['"); !r) {
    return r;
  }
  if (c.peek() == ']') {
    c.eat(']');
    return {};
  }
  for (;;) {
    if (auto r = expect(c, '{', "metadata entry"); !r) {
      return r;
    }
    std::string key;
    std::string t = "s";
    std::string v;
    if (c.peek() != '}') {
      for (;;) {
        auto k = parse_json_string(c);
        if (!k) {
          return std::unexpected(k.error());
        }
        if (auto r = expect(c, ':', "':'"); !r) {
          return r;
        }
        if (*k == "key") {
          auto s = parse_json_string(c);
          if (!s) {
            return std::unexpected(s.error());
          }
          key = std::move(*s);
        } else if (*k == "t") {
          auto s = parse_json_string(c);
          if (!s) {
            return std::unexpected(s.error());
          }
          t = std::move(*s);
        } else if (*k == "v") {
          auto s = parse_json_string(c);
          if (!s) {
            return std::unexpected(s.error());
          }
          v = std::move(*s);
        } else {
          if (auto r = skip_json_value(c); !r) {
            return r;
          }
        }
        if (c.eat('}')) {
          break;
        }
        if (auto r = expect(c, ',', "',' or '}'"); !r) {
          return r;
        }
      }
    } else {
      c.eat('}');
    }
    if (!key.empty() && key != "nombrilo.operators" && key != "nombrilo.offsets" &&
        key != "nombrilo.children") {
      if (t == "i") {
        char *end = nullptr;
        const auto n = std::strtoll(v.c_str(), &end, 10);
        if (end != v.c_str()) {
          if (auto r = b.set(key, static_cast<std::int64_t>(n)); !r) {
            return r;
          }
        }
      } else if (t == "f") {
        char *end = nullptr;
        const auto n = std::strtod(v.c_str(), &end);
        if (end != v.c_str()) {
          if (auto r = b.set(key, n); !r) {
            return r;
          }
        }
      } else if (t == "b") {
        auto bytes = hex_decode(v);
        if (!bytes) {
          return std::unexpected(bytes.error());
        }
        if (auto r = b.set(key, std::span<const std::byte>(*bytes)); !r) {
          return r;
        }
      } else {
        if (auto r = b.set(key, v); !r) {
          return r;
        }
      }
    }
    if (c.eat(']')) {
      return {};
    }
    if (auto r = expect(c, ',', "',' or ']'"); !r) {
      return r;
    }
  }
}

struct QuerySpec {
  std::uint32_t lit = 1;
  std::string kind = "PROBABILITY";
  std::string name;
  std::vector<double> confidence;
};

std::string dump_queries_json(std::span<const QuerySpec> qs) {
  std::string out = "[";
  for (std::size_t i = 0; i < qs.size(); ++i) {
    if (i) {
      out += ',';
    }
    out += "{\"lit\":";
    out += std::to_string(qs[i].lit);
    out += ",\"kind\":";
    out += json_string(qs[i].kind);
    out += ",\"name\":";
    out += json_string(qs[i].name);
    out += ",\"confidence\":[";
    for (std::size_t j = 0; j < qs[i].confidence.size(); ++j) {
      if (j) {
        out += ',';
      }
      std::ostringstream os;
      os << std::setprecision(17) << qs[i].confidence[j];
      out += os.str();
    }
    out += "]}";
  }
  out += ']';
  return out;
}

pcfb::Result<std::vector<QuerySpec>> parse_queries_json(std::string_view json) {
  JsonCursor c{json, 0};
  if (auto r = expect(c, '[', "'['"); !r) {
    return std::unexpected(r.error());
  }
  std::vector<QuerySpec> out;
  if (c.peek() == ']') {
    c.eat(']');
    return out;
  }
  for (;;) {
    if (auto r = expect(c, '{', "query object"); !r) {
      return std::unexpected(r.error());
    }
    QuerySpec q;
    if (c.peek() != '}') {
      for (;;) {
        auto key = parse_json_string(c);
        if (!key) {
          return std::unexpected(key.error());
        }
        if (auto r = expect(c, ':', "':'"); !r) {
          return std::unexpected(r.error());
        }
        if (*key == "lit" || *key == "target") {
          auto n = parse_json_u64(c);
          if (!n) {
            return std::unexpected(n.error());
          }
          if (*n > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected(nombrilo_error("query lit out of range"));
          }
          q.lit = static_cast<std::uint32_t>(*n);
        } else if (*key == "kind") {
          auto s = parse_json_string(c);
          if (!s) {
            return std::unexpected(s.error());
          }
          q.kind = std::move(*s);
        } else if (*key == "name") {
          auto s = parse_json_string(c);
          if (!s) {
            return std::unexpected(s.error());
          }
          q.name = std::move(*s);
        } else if (*key == "confidence") {
          if (auto r = expect(c, '[', "'['"); !r) {
            return std::unexpected(r.error());
          }
          if (c.peek() != ']') {
            for (;;) {
              c.skip();
              char *end = nullptr;
              const std::string rest(c.s.substr(c.i));
              const double d = std::strtod(rest.c_str(), &end);
              if (end == rest.c_str()) {
                return std::unexpected(nombrilo_error("query confidence is not a number"));
              }
              c.i += static_cast<std::size_t>(end - rest.c_str());
              q.confidence.push_back(d);
              if (c.eat(']')) {
                break;
              }
              if (auto r = expect(c, ',', "',' or ']'"); !r) {
                return std::unexpected(r.error());
              }
            }
          } else {
            c.eat(']');
          }
        } else {
          if (auto r = skip_json_value(c); !r) {
            return std::unexpected(r.error());
          }
        }
        if (c.eat('}')) {
          break;
        }
        if (auto r = expect(c, ',', "',' or '}'"); !r) {
          return std::unexpected(r.error());
        }
      }
    } else {
      c.eat('}');
    }
    out.push_back(std::move(q));
    if (c.eat(']')) {
      return out;
    }
    if (auto r = expect(c, ',', "',' or ']'"); !r) {
      return std::unexpected(r.error());
    }
  }
}

pcfb::SampleSource source_from_name(std::string_view s) {
  if (s == "COLLAPSED_MODULE") {
    return pcfb::SampleSource::COLLAPSED_MODULE;
  }
  if (s == "HOUSE_EVENT") {
    return pcfb::SampleSource::HOUSE_EVENT;
  }
  return pcfb::SampleSource::BASIC_EVENT;
}

const char *source_to_name(pcfb::SampleSource s) {
  switch (s) {
    case pcfb::SampleSource::COLLAPSED_MODULE:
      return "COLLAPSED_MODULE";
    case pcfb::SampleSource::HOUSE_EVENT:
      return "HOUSE_EVENT";
    default:
      return "BASIC_EVENT";
  }
}

std::string dump_sources_json(std::span<const pcfb::SampleSource> xs) {
  std::string out = "[";
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (i) {
      out += ',';
    }
    out += json_string(source_to_name(xs[i]));
  }
  out += ']';
  return out;
}

pcfb::Result<std::vector<pcfb::SampleSource>> parse_sources_json(std::string_view json) {
  JsonCursor c{json, 0};
  if (auto r = expect(c, '[', "'['"); !r) {
    return std::unexpected(r.error());
  }
  std::vector<pcfb::SampleSource> out;
  if (c.peek() == ']') {
    c.eat(']');
    return out;
  }
  for (;;) {
    auto s = parse_json_string(c);
    if (!s) {
      return std::unexpected(s.error());
    }
    out.push_back(source_from_name(*s));
    if (c.eat(']')) {
      return out;
    }
    if (auto r = expect(c, ',', "',' or ']'"); !r) {
      return std::unexpected(r.error());
    }
  }
}

std::string unique_name(std::string base, std::unordered_set<std::string> &used) {
  if (base.empty()) {
    base = "n";
  }
  if (used.insert(base).second) {
    return base;
  }
  for (std::uint64_t i = 2;; ++i) {
    auto cand = base + "_" + std::to_string(i);
    if (used.insert(cand).second) {
      return cand;
    }
  }
}

}  // namespace

bool looks_like_nombrilo(std::span<const std::byte> bytes) {
  if (bytes.size() < 8) {
    return false;
  }
  const auto n = read_u64_le(bytes.data());
  if (n == 0 || n > kMaxHeaderBytes || n > bytes.size() - 8) {
    return false;
  }
  const auto json = std::string_view(reinterpret_cast<const char *>(bytes.data() + 8), static_cast<std::size_t>(n));
  auto header = parse_safetensors_header(json);
  if (!header) {
    return false;
  }
  auto it = header->metadata.find("format");
  return it != header->metadata.end() && it->second == kNombriloFormat;
}

bool looks_like_nombrilo_file(const std::string &path) {
  auto bytes = load_file(path);
  return bytes && looks_like_nombrilo(*bytes);
}

pcfb::Result<NombriloCircuit> read_nombrilo(std::span<const std::byte> bytes) {
  if (bytes.size() < 8) {
    return std::unexpected(nombrilo_error("file too small to be safetensors"));
  }
  const auto n = read_u64_le(bytes.data());
  if (n == 0 || n > kMaxHeaderBytes) {
    return std::unexpected(nombrilo_error("safetensors header length is invalid"));
  }
  if (n > bytes.size() - 8) {
    return std::unexpected(nombrilo_error("truncated safetensors header"));
  }
  const auto json = std::string_view(reinterpret_cast<const char *>(bytes.data() + 8), static_cast<std::size_t>(n));
  auto header = parse_safetensors_header(json);
  if (!header) {
    return std::unexpected(header.error());
  }
  const auto format = header->metadata["format"];
  if (format != kNombriloFormat) {
    return std::unexpected(nombrilo_error("unsupported format '" + format + "' (expected nombrilo.circuit)"));
  }
  const auto version = header->metadata["version"];
  if (version != kNombriloVersion) {
    return std::unexpected(nombrilo_error("unsupported nombrilo version '" + version + "' (expected 1)"));
  }
  const auto bc = header->metadata["base_count"];
  if (bc.empty()) {
    return std::unexpected(nombrilo_error("missing base_count"));
  }
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(bc.c_str(), &end, 10);
  if (end == bc.c_str() || *end != '\0' || parsed > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected(nombrilo_error("invalid base_count"));
  }

  auto data = bytes.subspan(8 + static_cast<std::size_t>(n));
  auto need = [&](const char *name) -> pcfb::Result<TensorInfo> {
    auto it = header->tensors.find(name);
    if (it == header->tensors.end()) {
      return std::unexpected(nombrilo_error(std::string("missing tensor '") + name + "'"));
    }
    return it->second;
  };

  NombriloCircuit c;
  c.base_count = static_cast<std::uint32_t>(parsed);
  c.metadata = std::move(header->metadata);

  auto ops_info = need("operators");
  if (!ops_info) {
    return std::unexpected(ops_info.error());
  }
  auto off_info = need("offsets");
  if (!off_info) {
    return std::unexpected(off_info.error());
  }
  auto ch_info = need("children");
  if (!ch_info) {
    return std::unexpected(ch_info.error());
  }
  auto operators = load_u32(data, *ops_info, "operators");
  if (!operators) {
    return std::unexpected(operators.error());
  }
  auto offsets = load_u32(data, *off_info, "offsets");
  if (!offsets) {
    return std::unexpected(offsets.error());
  }
  auto children = load_u32(data, *ch_info, "children");
  if (!children) {
    return std::unexpected(children.error());
  }
  c.operators = std::move(*operators);
  c.offsets = std::move(*offsets);
  c.children = std::move(*children);

  const bool have_names = header->tensors.contains("names");
  const bool have_lits = header->tensors.contains("name_lits");
  if (have_names != have_lits) {
    return std::unexpected(nombrilo_error("names and name_lits must both be present or both absent"));
  }
  if (have_names) {
    const auto &ni = header->tensors["names"];
    if (!dtype_is(ni.dtype, "U8", "u8")) {
      return std::unexpected(nombrilo_error("names dtype must be U8"));
    }
    auto raw = tensor_bytes(data, ni, "names", 1);
    if (!raw) {
      return std::unexpected(raw.error());
    }
    auto names = split_names(*raw);
    if (!names) {
      return std::unexpected(names.error());
    }
    auto lits = load_u32(data, header->tensors["name_lits"], "name_lits");
    if (!lits) {
      return std::unexpected(lits.error());
    }
    c.names = std::move(*names);
    c.name_lits = std::move(*lits);
  }
  if (header->tensors.contains("weights")) {
    auto w = load_f64(data, header->tensors["weights"], "weights");
    if (!w) {
      return std::unexpected(w.error());
    }
    c.weights = std::move(*w);
  }
  if (auto r = validate_circuit(c); !r) {
    return std::unexpected(r.error());
  }
  return c;
}

pcfb::Result<NombriloCircuit> read_nombrilo_file(const std::string &path) {
  auto bytes = load_file(path);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  return read_nombrilo(*bytes);
}

pcfb::Result<std::vector<std::byte>> write_nombrilo(const NombriloCircuit &circuit) {
  if (auto r = validate_circuit(circuit); !r) {
    return std::unexpected(r.error());
  }

  std::map<std::string, std::string> meta = circuit.metadata;
  meta["format"] = kNombriloFormat;
  meta["version"] = kNombriloVersion;
  meta["base_count"] = std::to_string(circuit.base_count);

  struct Packed {
    std::string name;
    std::string dtype;
    std::uint64_t numel = 0;
    std::vector<std::byte> bytes;
  };
  std::vector<Packed> tensors;
  auto add_u32 = [&](const char *name, std::span<const std::uint32_t> xs) {
    Packed p;
    p.name = name;
    p.dtype = "U32";
    p.numel = xs.size();
    p.bytes.reserve(xs.size() * 4);
    for (auto v : xs) {
      write_u32_le(p.bytes, v);
    }
    tensors.push_back(std::move(p));
  };

  if (circuit.weights) {
    Packed p;
    p.name = "weights";
    p.dtype = "F64";
    p.numel = circuit.weights->size();
    p.bytes.reserve(circuit.weights->size() * 8);
    for (double v : *circuit.weights) {
      write_f64_le(p.bytes, v);
    }
    tensors.push_back(std::move(p));
  }
  add_u32("children", circuit.children);
  if (!circuit.names.empty() || !circuit.name_lits.empty()) {
    add_u32("name_lits", circuit.name_lits);
  }
  add_u32("offsets", circuit.offsets);
  add_u32("operators", circuit.operators);
  if (!circuit.names.empty() || !circuit.name_lits.empty()) {
    Packed p;
    p.name = "names";
    p.dtype = "U8";
    std::vector<std::byte> raw;
    for (std::size_t i = 0; i < circuit.names.size(); ++i) {
      if (i) {
        raw.push_back(std::byte{0});
      }
      for (unsigned char ch : circuit.names[i]) {
        raw.push_back(static_cast<std::byte>(ch));
      }
    }
    p.numel = raw.size();
    p.bytes = std::move(raw);
    tensors.push_back(std::move(p));
  }

  std::uint64_t cursor = 0;
  std::string header = "{\"__metadata__\":{";
  bool first_meta = true;
  auto emit_meta = [&](std::string_view key) {
    auto it = meta.find(std::string(key));
    if (it == meta.end()) {
      return;
    }
    if (!first_meta) {
      header += ',';
    }
    first_meta = false;
    header += json_string(it->first);
    header += ':';
    header += json_string(it->second);
  };
  // Stable core key order, then any extras in map order.
  emit_meta("format");
  emit_meta("version");
  emit_meta("base_count");
  for (const auto &[k, v] : meta) {
    if (k == "format" || k == "version" || k == "base_count") {
      continue;
    }
    if (!first_meta) {
      header += ',';
    }
    first_meta = false;
    header += json_string(k);
    header += ':';
    header += json_string(v);
  }
  header += '}';
  for (auto &t : tensors) {
    header += ',';
    header += json_string(t.name);
    header += ':';
    header += tensor_json(t.dtype, t.numel, cursor, cursor + t.bytes.size());
    cursor += t.bytes.size();
  }
  header += '}';
  while ((8 + header.size()) % 8 != 0) {
    header.push_back(' ');
  }

  std::vector<std::byte> out;
  out.reserve(8 + header.size() + cursor);
  write_u64_le(out, header.size());
  for (unsigned char ch : header) {
    out.push_back(static_cast<std::byte>(ch));
  }
  for (const auto &t : tensors) {
    out.insert(out.end(), t.bytes.begin(), t.bytes.end());
  }
  return out;
}

pcfb::Result<void> inherit_nombrilo_weights(NombriloCircuit &circuit, const std::string &source_path,
                                            const std::optional<std::string> &weights_path) {
  if (circuit.weights) {
    return {};
  }
  std::string path;
  if (weights_path && !weights_path->empty()) {
    path = *weights_path;
  } else {
    auto sib = sibling_base_path(source_path);
    if (!sib) {
      return std::unexpected(
          nombrilo_error("variant file has no weights and no sibling base <model>.safetensors was found"));
    }
    path = *sib;
  }
  auto base = read_nombrilo_file(path);
  if (!base) {
    return std::unexpected(base.error());
  }
  if (base->base_count != circuit.base_count) {
    return std::unexpected(nombrilo_error("weights file base_count " + std::to_string(base->base_count) +
                                          " does not match variant base_count " +
                                          std::to_string(circuit.base_count)));
  }
  if (input_name_table(*base) != input_name_table(circuit)) {
    return std::unexpected(nombrilo_error("weights file input names do not match the variant"));
  }
  if (!base->weights) {
    return std::unexpected(nombrilo_error("weights file " + path + " itself has no weights"));
  }
  circuit.weights = base->weights;
  circuit.metadata["nombrilo.weights_source"] = "inherited";
  circuit.metadata["nombrilo.weights_file"] = std::filesystem::path(path).filename().string();
  return {};
}

pcfb::Result<Compiled> nombrilo_to_pcfb(const NombriloCircuit &src, const NombriloCompileOptions &options,
                                        std::span<const std::byte> source_sha256, std::string_view model_name,
                                        std::string_view source_path) {
  if (auto r = validate_circuit(src); !r) {
    return std::unexpected(r.error());
  }
  NombriloCircuit circuit = src;
  const bool inherited = meta_get(circuit, "nombrilo.weights_source") == "inherited";
  if (!circuit.weights) {
    return std::unexpected(nombrilo_error("nombrilo circuit has no weights (inherit them first)"));
  }
  if (circuit.weights->size() != circuit.base_count) {
    return std::unexpected(nombrilo_error("weights length must equal base_count"));
  }

  const auto B = circuit.base_count;
  const auto G = circuit.operators.size();

  pcfb::builder b;
  b.name(model_name);

  auto ones = b.add_op(pcfb::OpCode::BITWISE_ONES, {}, {}, std::nullopt, "true");
  if (!ones) {
    return std::unexpected(ones.error());
  }
  if (*ones != 0) {
    return std::unexpected(nombrilo_error("internal: constant TRUE is not tensor 0"));
  }

  for (std::uint32_t v = 1; v <= B; ++v) {
    const double p = (*circuit.weights)[v - 1];
    auto t = b.add_sample(p, first_positive_name(circuit, v), v, pcfb::SampleSource::BASIC_EVENT);
    if (!t) {
      return std::unexpected(t.error());
    }
    if (*t != v) {
      return std::unexpected(nombrilo_error("internal: SAMPLE tensor index mismatch"));
    }
  }

  for (std::size_t i = 0; i < G; ++i) {
    const auto lo = circuit.offsets[i];
    const auto hi = circuit.offsets[i + 1];
    const auto n = hi - lo;
    const auto code = circuit.operators[i];
    const auto gate_var = B + 1 + static_cast<std::uint32_t>(i);
    const auto gname = first_positive_name(circuit, gate_var);

    std::vector<std::uint32_t> inputs;
    std::vector<std::uint32_t> inverted;
    for (auto j = lo; j < hi; ++j) {
      const auto lit = circuit.children[j];
      const auto tv = lit_var(lit);
      if (lit_pos(lit)) {
        inputs.push_back(tv);
      } else {
        inverted.push_back(tv);
      }
    }

    auto emit_const = [&](pcfb::OpCode op) -> pcfb::Result<pcfb::builder::tensor_id> {
      return b.add_op(op, {}, {}, std::nullopt, gname);
    };

    pcfb::Result<pcfb::builder::tensor_id> t = std::unexpected(nombrilo_error("internal"));
    if (code == kNombriloAnd) {
      if (n == 0) {
        t = emit_const(pcfb::OpCode::BITWISE_ONES);
      } else {
        t = b.add_op(pcfb::OpCode::BITWISE_AND, inputs, inverted, std::nullopt, gname);
      }
    } else if (code == kNombriloOr) {
      if (n == 0) {
        t = emit_const(pcfb::OpCode::BITWISE_ZEROS);
      } else {
        t = b.add_op(pcfb::OpCode::BITWISE_OR, inputs, inverted, std::nullopt, gname);
      }
    } else if (code == kXorCode) {
      if (n == 0) {
        t = emit_const(pcfb::OpCode::BITWISE_ZEROS);
      } else {
        t = b.add_op(pcfb::OpCode::BITWISE_XOR, inputs, inverted, std::nullopt, gname);
      }
    } else {
      // at-least-k
      if (n == 0) {
        t = emit_const(code == 0 ? pcfb::OpCode::BITWISE_ONES : pcfb::OpCode::BITWISE_ZEROS);
      } else if (code > n) {
        t = emit_const(pcfb::OpCode::BITWISE_ZEROS);
      } else {
        t = b.add_op(pcfb::OpCode::BITWISE_ATLEAST, inputs, inverted, code, gname);
      }
    }
    if (!t) {
      return std::unexpected(t.error());
    }
    if (*t != gate_var) {
      return std::unexpected(nombrilo_error("internal: gate tensor index mismatch (lowered constant?)"));
    }
  }

  auto add_query_lit = [&](std::uint32_t lit, std::string name) -> pcfb::Result<void> {
    auto target = lit_var(lit);
    if (!lit_pos(lit)) {
      const std::array<std::uint32_t, 1> ins{target};
      auto neg = b.add_op(pcfb::OpCode::BITWISE_NOT, ins, {}, std::nullopt, name.empty() ? "not" : name);
      if (!neg) {
        return std::unexpected(neg.error());
      }
      target = *neg;
    }
    if (name.empty()) {
      name = "t" + std::to_string(target);
    }
    return b.add_query(target, pcfb::QueryKind::PROBABILITY, name);
  };

  if (auto extras = meta_get(circuit, "fuzzybits.pcfb.queries"); !extras.empty()) {
    auto qs = parse_queries_json(extras);
    if (!qs) {
      return std::unexpected(qs.error());
    }
    for (auto &q : *qs) {
      if (auto r = add_query_lit(q.lit, q.name); !r) {
        return std::unexpected(r.error());
      }
    }
  } else if (options.target) {
    auto lit = lookup_name(circuit, *options.target);
    if (!lit) {
      return std::unexpected(nombrilo_error("target name '" + *options.target + "' not found"));
    }
    if (auto r = add_query_lit(*lit, *options.target); !r) {
      return std::unexpected(r.error());
    }
  } else {
    std::vector<char> has_parent(B + 1 + G, 0);
    for (auto lit : circuit.children) {
      const auto v = lit_var(lit);
      if (v < has_parent.size()) {
        has_parent[v] = 1;
      }
    }
    std::size_t nq = 0;
    for (std::size_t i = 0; i < G; ++i) {
      const auto v = B + 1 + static_cast<std::uint32_t>(i);
      if (has_parent[v]) {
        continue;
      }
      auto qname = first_positive_name(circuit, v);
      if (qname.empty()) {
        qname = "g" + std::to_string(i);
      }
      if (auto r = add_query_lit(make_lit(v, true), qname); !r) {
        return std::unexpected(r.error());
      }
      ++nq;
    }
    if (nq == 0) {
      return std::unexpected(nombrilo_error("circuit has no parent-free gate to query"));
    }
  }

  if (auto r = set_meta(b, "general.generator", "fuzzybits"); !r) {
    return std::unexpected(r.error());
  }
  if (auto r = set_meta(b, "general.generator_version", generator_version()); !r) {
    return std::unexpected(r.error());
  }
  if (auto r = set_meta(b, "general.model_name", model_name); !r) {
    return std::unexpected(r.error());
  }
  if (meta_get(circuit, "fuzzybits.pcfb.metadata").empty()) {
    if (auto r = set_meta(b, "general.source_format", kNombriloSourceFormat); !r) {
      return std::unexpected(r.error());
    }
  }
  if (!source_sha256.empty()) {
    if (auto r = b.set("general.source_sha256", source_sha256); !r) {
      return std::unexpected(r.error());
    }
  }
  if (!source_path.empty()) {
    if (auto r = set_meta(b, "fuzzybits.source", source_path); !r) {
      return std::unexpected(r.error());
    }
  }
  if (auto r = write_nombrilo_side_meta(b, circuit, inherited); !r) {
    return std::unexpected(r.error());
  }

  if (auto extras = meta_get(circuit, "fuzzybits.pcfb.event_ids"); !extras.empty()) {
    auto ids = parse_u32_array_text(extras);
    if (!ids) {
      return std::unexpected(ids.error());
    }
    // SAMPLE ops already used event_id = v. Restoring a different event_id
    // requires a second builder pass; record the requested ids for the
    // export round-trip and rebuild samples if they differ.
    (void)ids;
  }
  if (auto blob = meta_get(circuit, "fuzzybits.pcfb.metadata"); !blob.empty()) {
    if (auto r = restore_pcfb_metadata(b, blob); !r) {
      return std::unexpected(r.error());
    }
  }

  auto bytes = b.finish();
  if (!bytes) {
    return std::unexpected(bytes.error());
  }

  // If extras request different event_ids / sources, rebuild so SAMPLE args match.
  const auto event_json = meta_get(circuit, "fuzzybits.pcfb.event_ids");
  const auto source_json = meta_get(circuit, "fuzzybits.pcfb.sources");
  if (!event_json.empty() || !source_json.empty()) {
    auto ids = event_json.empty() ? pcfb::Result<std::vector<std::uint32_t>>(std::vector<std::uint32_t>{})
                                  : parse_u32_array_text(event_json);
    if (!ids) {
      return std::unexpected(ids.error());
    }
    auto sources = source_json.empty() ? pcfb::Result<std::vector<pcfb::SampleSource>>(std::vector<pcfb::SampleSource>{})
                                       : parse_sources_json(source_json);
    if (!sources) {
      return std::unexpected(sources.error());
    }
    if ((!ids->empty() && ids->size() != B) || (!sources->empty() && sources->size() != B)) {
      return std::unexpected(nombrilo_error("fuzzybits.pcfb.event_ids/sources length must equal base_count"));
    }
    const bool need_rebuild =
        (!ids->empty() && [&] {
          for (std::uint32_t v = 1; v <= B; ++v) {
            if ((*ids)[v - 1] != v) {
              return true;
            }
          }
          return false;
        }()) ||
        (!sources->empty() && [&] {
          for (auto s : *sources) {
            if (s != pcfb::SampleSource::BASIC_EVENT) {
              return true;
            }
          }
          return false;
        }());
    if (need_rebuild) {
      pcfb::builder b2;
      b2.name(model_name);
      auto ones2 = b2.add_op(pcfb::OpCode::BITWISE_ONES, {}, {}, std::nullopt, "true");
      if (!ones2) {
        return std::unexpected(ones2.error());
      }
      std::unordered_set<std::uint32_t> used_ids;
      for (std::uint32_t v = 1; v <= B; ++v) {
        const auto eid = ids->empty() ? v : (*ids)[v - 1];
        if (!used_ids.insert(eid).second) {
          return std::unexpected(nombrilo_error("duplicate restored event_id"));
        }
        const auto src = sources->empty() ? pcfb::SampleSource::BASIC_EVENT : (*sources)[v - 1];
        auto t = b2.add_sample((*circuit.weights)[v - 1], first_positive_name(circuit, v), eid, src);
        if (!t) {
          return std::unexpected(t.error());
        }
      }
      for (std::size_t i = 0; i < G; ++i) {
        const auto lo = circuit.offsets[i];
        const auto hi = circuit.offsets[i + 1];
        const auto nch = hi - lo;
        const auto code = circuit.operators[i];
        const auto gate_var = B + 1 + static_cast<std::uint32_t>(i);
        const auto gname = first_positive_name(circuit, gate_var);
        std::vector<std::uint32_t> inputs;
        std::vector<std::uint32_t> inverted;
        for (auto j = lo; j < hi; ++j) {
          const auto lit = circuit.children[j];
          (lit_pos(lit) ? inputs : inverted).push_back(lit_var(lit));
        }
        pcfb::Result<pcfb::builder::tensor_id> t = std::unexpected(nombrilo_error("internal"));
        auto emit_const = [&](pcfb::OpCode op) { return b2.add_op(op, {}, {}, std::nullopt, gname); };
        if (code == kNombriloAnd) {
          t = nch == 0 ? emit_const(pcfb::OpCode::BITWISE_ONES)
                       : b2.add_op(pcfb::OpCode::BITWISE_AND, inputs, inverted, std::nullopt, gname);
        } else if (code == kNombriloOr) {
          t = nch == 0 ? emit_const(pcfb::OpCode::BITWISE_ZEROS)
                       : b2.add_op(pcfb::OpCode::BITWISE_OR, inputs, inverted, std::nullopt, gname);
        } else if (code == kXorCode) {
          t = nch == 0 ? emit_const(pcfb::OpCode::BITWISE_ZEROS)
                       : b2.add_op(pcfb::OpCode::BITWISE_XOR, inputs, inverted, std::nullopt, gname);
        } else if (nch == 0) {
          t = emit_const(code == 0 ? pcfb::OpCode::BITWISE_ONES : pcfb::OpCode::BITWISE_ZEROS);
        } else if (code > nch) {
          t = emit_const(pcfb::OpCode::BITWISE_ZEROS);
        } else {
          t = b2.add_op(pcfb::OpCode::BITWISE_ATLEAST, inputs, inverted, code, gname);
        }
        if (!t) {
          return std::unexpected(t.error());
        }
        if (*t != gate_var) {
          return std::unexpected(nombrilo_error("internal: rebuilt gate tensor mismatch"));
        }
      }
      auto add_q = [&](std::uint32_t lit, std::string name) -> pcfb::Result<void> {
        auto target = lit_var(lit);
        if (!lit_pos(lit)) {
          const std::array<std::uint32_t, 1> ins{target};
          auto neg = b2.add_op(pcfb::OpCode::BITWISE_NOT, ins, {}, std::nullopt, name.empty() ? "not" : name);
          if (!neg) {
            return std::unexpected(neg.error());
          }
          target = *neg;
        }
        return b2.add_query(target, pcfb::QueryKind::PROBABILITY, name);
      };
      if (auto extras = meta_get(circuit, "fuzzybits.pcfb.queries"); !extras.empty()) {
        auto qs = parse_queries_json(extras);
        if (!qs) {
          return std::unexpected(qs.error());
        }
        for (auto &q : *qs) {
          if (auto r = add_q(q.lit, q.name); !r) {
            return std::unexpected(r.error());
          }
        }
      } else if (options.target) {
        auto lit = lookup_name(circuit, *options.target);
        if (!lit) {
          return std::unexpected(nombrilo_error("target name '" + *options.target + "' not found"));
        }
        if (auto r = add_q(*lit, *options.target); !r) {
          return std::unexpected(r.error());
        }
      } else {
        std::vector<char> has_parent(B + 1 + G, 0);
        for (auto lit : circuit.children) {
          const auto v = lit_var(lit);
          if (v < has_parent.size()) {
            has_parent[v] = 1;
          }
        }
        for (std::size_t i = 0; i < G; ++i) {
          const auto v = B + 1 + static_cast<std::uint32_t>(i);
          if (has_parent[v]) {
            continue;
          }
          auto qname = first_positive_name(circuit, v);
          if (qname.empty()) {
            qname = "g" + std::to_string(i);
          }
          if (auto r = add_q(make_lit(v, true), qname); !r) {
            return std::unexpected(r.error());
          }
        }
      }
      if (auto r = set_meta(b2, "general.generator", "fuzzybits"); !r) {
        return std::unexpected(r.error());
      }
      if (auto r = set_meta(b2, "general.generator_version", generator_version()); !r) {
        return std::unexpected(r.error());
      }
      if (auto r = set_meta(b2, "general.model_name", model_name); !r) {
        return std::unexpected(r.error());
      }
      if (auto blob = meta_get(circuit, "fuzzybits.pcfb.metadata"); !blob.empty()) {
        if (auto r = restore_pcfb_metadata(b2, blob); !r) {
          return std::unexpected(r.error());
        }
      } else if (auto r = set_meta(b2, "general.source_format", kNombriloSourceFormat); !r) {
        return std::unexpected(r.error());
      }
      if (!source_sha256.empty()) {
        if (auto r = b2.set("general.source_sha256", source_sha256); !r) {
          return std::unexpected(r.error());
        }
      }
      if (auto r = write_nombrilo_side_meta(b2, circuit, inherited); !r) {
        return std::unexpected(r.error());
      }
      bytes = b2.finish();
      if (!bytes) {
        return std::unexpected(bytes.error());
      }
    }
  }

  Compiled out;
  out.bytes = std::move(*bytes);
  out.model_name = std::string(model_name);
  out.target_id = options.target.value_or("");
  out.target_name = out.target_id;
  if (out.target_id.empty()) {
    auto circuit_view = pcfb::io::from_bytes(out.bytes);
    if (circuit_view && circuit_view->subcircuits() && circuit_view->subcircuits()->size() > 0) {
      const auto *sc = circuit_view->subcircuits()->Get(0);
      if (sc && sc->queries() && sc->queries()->size() > 0 && sc->queries()->Get(0)->name()) {
        out.target_id = sc->queries()->Get(0)->name()->str();
        out.target_name = out.target_id;
      }
    }
  }
  return out;
}

pcfb::Result<Compiled> compile_nombrilo_file(const std::string &path, const NombriloCompileOptions &options) {
  auto circuit = read_nombrilo_file(path);
  if (!circuit) {
    return std::unexpected(circuit.error());
  }
  const bool had_weights = circuit->weights.has_value();
  if (auto r = inherit_nombrilo_weights(*circuit, path, options.weights_file); !r) {
    return std::unexpected(r.error());
  }
  if (!had_weights) {
    circuit->metadata["nombrilo.weights_source"] = "inherited";
  } else {
    circuit->metadata["nombrilo.weights_source"] = "inline";
  }
  std::array<std::uint8_t, 32> digest{};
  try {
    digest = sha256_file(path);
  } catch (const std::exception &ex) {
    return std::unexpected(io_error(ex.what()));
  }
  std::byte sha[32];
  for (int i = 0; i < 32; ++i) {
    sha[i] = static_cast<std::byte>(digest[i]);
  }
  auto model = std::filesystem::path(path).stem().string();
  const auto dot = model.rfind('.');
  if (dot != std::string::npos && is_variant_suffix(std::string_view(model).substr(dot + 1))) {
    // keep full stem (e.g. baobab1.e1) as the model name
  }
  return nombrilo_to_pcfb(*circuit, options, std::span<const std::byte>{sha, 32}, model, path);
}

namespace {

std::uint32_t opcode_to_nombrilo(pcfb::OpCode op, std::uint32_t k, bool *negate_out) {
  *negate_out = false;
  switch (op) {
    case pcfb::OpCode::BITWISE_AND:
      return kNombriloAnd;
    case pcfb::OpCode::BITWISE_NAND:
      *negate_out = true;
      return kNombriloAnd;
    case pcfb::OpCode::BITWISE_OR:
      return kNombriloOr;
    case pcfb::OpCode::BITWISE_NOR:
      *negate_out = true;
      return kNombriloOr;
    case pcfb::OpCode::BITWISE_XOR:
      return kXorCode;
    case pcfb::OpCode::BITWISE_XNOR:
      *negate_out = true;
      return kXorCode;
    case pcfb::OpCode::BITWISE_ATLEAST:
      return k;
    case pcfb::OpCode::BITWISE_ATMOST:
      *negate_out = true;
      return k + 1;
    default:
      return kNombriloAnd;
  }
}

bool is_1to1_gate_opcode(pcfb::OpCode op) {
  switch (op) {
    case pcfb::OpCode::BITWISE_AND:
    case pcfb::OpCode::BITWISE_OR:
    case pcfb::OpCode::BITWISE_XOR:
    case pcfb::OpCode::BITWISE_ATLEAST:
    case pcfb::OpCode::BITWISE_ONES:
    case pcfb::OpCode::BITWISE_ZEROS:
      return true;
    default:
      return false;
  }
}

void split_op_lits(const pcfb::Op *op, std::vector<std::uint32_t> &pos, std::vector<std::uint32_t> &neg) {
  pos.clear();
  neg.clear();
  if (op->inputs()) {
    for (auto id : *op->inputs()) {
      pos.push_back(make_lit(id, true));
    }
  }
  if (op->inverted_inputs()) {
    for (auto id : *op->inverted_inputs()) {
      neg.push_back(make_lit(id, false));
    }
  }
}

std::optional<NombriloCircuit> try_derive_1to1(const pcfb::ir::Circuit &circuit, const NombriloExportOptions &options) {
  if (!circuit.valid() || !circuit.subcircuits() || circuit.subcircuits()->size() == 0) {
    return std::nullopt;
  }
  const auto *sc = circuit.subcircuits()->Get(0);
  if (!sc || !sc->operators()) {
    return std::nullopt;
  }
  const auto *ops = sc->operators();
  if (!ops || ops->size() == 0) {
    return std::nullopt;
  }
  const auto *op0 = ops->Get(0);
  if (!op0 || op0->opcode() != pcfb::OpCode::BITWISE_ONES || op0->output() != 0) {
    return std::nullopt;
  }

  NombriloCircuit out;
  std::vector<double> weights;
  std::size_t i = 1;
  while (i < ops->size()) {
    const auto *op = ops->Get(i);
    if (!op || op->opcode() != pcfb::OpCode::SAMPLE || op->output() != i) {
      break;
    }
    const auto *sa = op->args_as_SampleArgs();
    if (!sa) {
      return std::nullopt;
    }
    weights.push_back(sa->probability());
    ++i;
  }
  const auto B = static_cast<std::uint32_t>(weights.size());
  out.base_count = B;

  std::size_t gate_begin = i;
  std::uint32_t expected = B + 1;
  while (i < ops->size()) {
    const auto *op = ops->Get(i);
    if (!op) {
      return std::nullopt;
    }
    const auto oc = op->opcode();
    if (oc == pcfb::OpCode::BITWISE_NOT || oc == pcfb::OpCode::BITWISE_NO_OP) {
      break;
    }
    if (!is_1to1_gate_opcode(oc) || op->output() != expected) {
      return std::nullopt;
    }
    ++expected;
    ++i;
  }
  const auto n_gates = static_cast<std::uint32_t>(i - gate_begin);
  while (i < ops->size()) {
    const auto *op = ops->Get(i);
    if (!op) {
      return std::nullopt;
    }
    const auto oc = op->opcode();
    if (oc != pcfb::OpCode::BITWISE_NOT && oc != pcfb::OpCode::BITWISE_NO_OP) {
      return std::nullopt;
    }
    ++i;
  }

  std::map<std::uint32_t, std::vector<std::uint32_t>> order_map;
  if (auto raw = pcfb_meta_string(circuit, "nombrilo.child_order")) {
    auto parsed = parse_child_order_object(*raw);
    if (!parsed) {
      child_order_warn(parsed.error().message);
    } else {
      order_map = std::move(*parsed);
    }
  }

  out.operators.reserve(n_gates);
  out.offsets.push_back(0);
  for (std::uint32_t g = 0; g < n_gates; ++g) {
    const auto *op = ops->Get(gate_begin + g);
    const auto oc = op->opcode();
    std::uint32_t code = kNombriloAnd;
    std::vector<std::uint32_t> pos;
    std::vector<std::uint32_t> neg;
    if (oc == pcfb::OpCode::BITWISE_ONES) {
      code = kNombriloAnd;
    } else if (oc == pcfb::OpCode::BITWISE_ZEROS) {
      code = kNombriloOr;
    } else {
      split_op_lits(op, pos, neg);
      if (oc == pcfb::OpCode::BITWISE_AND) {
        code = kNombriloAnd;
      } else if (oc == pcfb::OpCode::BITWISE_OR) {
        code = kNombriloOr;
      } else if (oc == pcfb::OpCode::BITWISE_XOR) {
        code = kXorCode;
      } else {
        const auto *kn = op->args_as_KofNArgs();
        code = kn ? kn->k() : 0;
      }
    }
    const std::vector<std::uint32_t> *pol = nullptr;
    if (auto it = order_map.find(g); it != order_map.end()) {
      pol = &it->second;
    }
    auto kids = apply_child_order(pos, neg, pol, g);
    out.operators.push_back(code);
    out.children.insert(out.children.end(), kids.begin(), kids.end());
    out.offsets.push_back(static_cast<std::uint32_t>(out.children.size()));
  }

  if (auto names_json = pcfb_meta_string(circuit, "nombrilo.names")) {
    auto names = parse_names_blob(*names_json);
    if (!names) {
      return std::nullopt;
    }
    out.names = std::move(names->names);
    out.name_lits = std::move(names->lits);
  } else {
    std::unordered_set<std::string> used;
    auto add = [&](std::string name, std::uint32_t lit) {
      if (name.empty()) {
        return;
      }
      name = unique_name(std::move(name), used);
      out.names.push_back(std::move(name));
      out.name_lits.push_back(lit);
    };
    for (std::size_t s = 1; s <= B; ++s) {
      const auto *op = ops->Get(s);
      const auto *sa = op->args_as_SampleArgs();
      if (sa && sa->name()) {
        add(sa->name()->str(), make_lit(static_cast<std::uint32_t>(s), true));
      }
    }
    for (std::uint32_t g = 0; g < n_gates; ++g) {
      const auto *op = ops->Get(gate_begin + g);
      if (op->name() && op->name()->size()) {
        add(op->name()->str(), make_lit(B + 1 + g, true));
      }
    }
  }

  const auto wsrc = pcfb_meta_string(circuit, "nombrilo.weights_source").value_or("inline");
  if (wsrc != "inherited") {
    out.weights = std::move(weights);
  }
  out.metadata["format"] = kNombriloFormat;
  out.metadata["version"] = kNombriloVersion;
  out.metadata["base_count"] = std::to_string(out.base_count);
  if (options.extras) {
    for (const char *key : {"fuzzybits.pcfb.queries", "fuzzybits.pcfb.event_ids", "fuzzybits.pcfb.sources",
                            "fuzzybits.pcfb.metadata"}) {
      if (auto v = pcfb_meta_string(circuit, key)) {
        out.metadata[key] = *v;
      }
    }
  }
  if (auto r = validate_circuit(out); !r) {
    return std::nullopt;
  }
  return out;
}

}  // namespace

pcfb::Result<NombriloCircuit> pcfb_to_nombrilo(const pcfb::ir::Circuit &circuit, const NombriloExportOptions &options) {
  if (!circuit.valid() || !circuit.subcircuits() || circuit.subcircuits()->size() == 0) {
    return std::unexpected(nombrilo_error("empty pcfb circuit"));
  }
  const auto *sc = circuit.subcircuits()->Get(0);
  if (!sc || !sc->operators() || !sc->tensors()) {
    return std::unexpected(nombrilo_error("pcfb subcircuit missing operators"));
  }

  if (auto derived = try_derive_1to1(circuit, options)) {
    return std::move(*derived);
  }

  NombriloCircuit out;
  const auto n_tensors = sc->tensors() ? sc->tensors()->size() : 0;
  std::vector<std::optional<std::uint32_t>> lit_of(n_tensors);
  std::vector<std::uint32_t> sample_order;
  std::vector<double> sample_p;
  std::vector<std::uint32_t> sample_event;
  std::vector<pcfb::SampleSource> sample_source;
  std::vector<std::string> sample_name;

  for (const auto *op : *sc->operators()) {
    if (!op || op->opcode() != pcfb::OpCode::SAMPLE) {
      continue;
    }
    const auto *sa = op->args_as_SampleArgs();
    if (!sa) {
      return std::unexpected(nombrilo_error("SAMPLE without SampleArgs"));
    }
    const auto t = op->output();
    if (t >= n_tensors) {
      return std::unexpected(nombrilo_error("SAMPLE output out of range"));
    }
    sample_order.push_back(t);
    sample_p.push_back(sa->probability());
    sample_event.push_back(sa->event_id());
    sample_source.push_back(sa->source());
    sample_name.push_back(sa->name() ? sa->name()->str() : "");
  }

  out.base_count = static_cast<std::uint32_t>(sample_order.size());
  out.weights = sample_p;
  std::unordered_map<std::uint32_t, std::uint32_t> sample_var;
  for (std::size_t i = 0; i < sample_order.size(); ++i) {
    sample_var[sample_order[i]] = static_cast<std::uint32_t>(i + 1);
  }

  std::unordered_set<std::string> used_names;
  auto add_name = [&](std::string name, std::uint32_t lit) {
    if (name.empty()) {
      return;
    }
    name = unique_name(std::move(name), used_names);
    out.names.push_back(std::move(name));
    out.name_lits.push_back(lit);
  };
  for (std::size_t i = 0; i < sample_order.size(); ++i) {
    add_name(sample_name[i], make_lit(static_cast<std::uint32_t>(i + 1), true));
  }

  auto require_lit = [&](std::uint32_t t, bool invert) -> pcfb::Result<std::uint32_t> {
    if (t >= lit_of.size() || !lit_of[t]) {
      return std::unexpected(nombrilo_error("tensor " + std::to_string(t) + " used before it is defined"));
    }
    auto lit = *lit_of[t];
    if (invert) {
      lit ^= 1u;
    }
    return lit;
  };

  std::vector<std::uint32_t> gate_n;
  auto emit_gate = [&](std::uint32_t code, std::span<const std::uint32_t> kids) -> std::uint32_t {
    const auto g = out.base_count + 1 + static_cast<std::uint32_t>(out.operators.size());
    out.operators.push_back(code);
    gate_n.push_back(static_cast<std::uint32_t>(kids.size()));
    out.children.insert(out.children.end(), kids.begin(), kids.end());
    return g;
  };

  for (const auto *op : *sc->operators()) {
    if (!op) {
      continue;
    }
    const auto t = op->output();
    if (t >= lit_of.size()) {
      lit_of.resize(t + 1);
    }
    const auto opcode = op->opcode();
    if (opcode == pcfb::OpCode::SAMPLE) {
      lit_of[t] = make_lit(sample_var[t], true);
      continue;
    }
    if (opcode == pcfb::OpCode::BITWISE_ONES) {
      lit_of[t] = make_lit(0, true);
      if (op->name() && op->name()->size() && op->name()->str() != "true") {
        add_name(op->name()->str(), make_lit(0, true));
      }
      continue;
    }
    if (opcode == pcfb::OpCode::BITWISE_ZEROS) {
      lit_of[t] = make_lit(0, false);
      if (op->name() && op->name()->size()) {
        add_name(op->name()->str(), make_lit(0, false));
      }
      continue;
    }
    if (opcode == pcfb::OpCode::BITWISE_NO_OP || opcode == pcfb::OpCode::BITWISE_NOT) {
      std::uint32_t in = 0;
      bool inv = opcode == pcfb::OpCode::BITWISE_NOT;
      if (op->inputs() && op->inputs()->size()) {
        in = op->inputs()->Get(0);
      } else if (op->inverted_inputs() && op->inverted_inputs()->size()) {
        in = op->inverted_inputs()->Get(0);
        inv = !inv;
      } else {
        return std::unexpected(nombrilo_error("unary op has no input"));
      }
      auto lit = require_lit(in, inv);
      if (!lit) {
        return std::unexpected(lit.error());
      }
      lit_of[t] = *lit;
      if (op->name() && op->name()->size()) {
        add_name(op->name()->str(), *lit);
      }
      continue;
    }

    auto collect_lits = [&]() -> pcfb::Result<std::vector<std::uint32_t>> {
      std::vector<std::uint32_t> lits;
      if (op->inputs()) {
        for (auto id : *op->inputs()) {
          auto lit = require_lit(id, false);
          if (!lit) {
            return std::unexpected(lit.error());
          }
          lits.push_back(*lit);
        }
      }
      if (op->inverted_inputs()) {
        for (auto id : *op->inverted_inputs()) {
          auto lit = require_lit(id, true);
          if (!lit) {
            return std::unexpected(lit.error());
          }
          lits.push_back(*lit);
        }
      }
      return lits;
    };

    auto kids = collect_lits();
    if (!kids) {
      return std::unexpected(kids.error());
    }

    switch (opcode) {
      case pcfb::OpCode::BITWISE_AND:
      case pcfb::OpCode::BITWISE_NAND:
      case pcfb::OpCode::BITWISE_OR:
      case pcfb::OpCode::BITWISE_NOR:
      case pcfb::OpCode::BITWISE_XOR:
      case pcfb::OpCode::BITWISE_XNOR:
      case pcfb::OpCode::BITWISE_ATLEAST:
      case pcfb::OpCode::BITWISE_ATMOST:
      case pcfb::OpCode::BITWISE_EXACTLY:
        break;
      default:
        return std::unexpected(nombrilo_error("pcfb opcode cannot be represented in nombrilo"));
    }

    if (opcode == pcfb::OpCode::BITWISE_EXACTLY) {
      const std::uint32_t k =
          (op->args_type() == pcfb::OpArgs::KofNArgs && op->args_as_KofNArgs()) ? op->args_as_KofNArgs()->k() : 0;
      const auto g1 = emit_gate(k, *kids);
      const auto g2 = emit_gate(k + 1, *kids);
      const std::array<std::uint32_t, 2> and_kids{make_lit(g1, true), make_lit(g2, false)};
      const auto g3 = emit_gate(kNombriloAnd, and_kids);
      lit_of[t] = make_lit(g3, true);
      if (op->name() && op->name()->size()) {
        add_name(op->name()->str(), make_lit(g3, true));
      }
      continue;
    }

    bool negate = false;
    std::uint32_t k = 0;
    if (op->args_type() == pcfb::OpArgs::KofNArgs && op->args_as_KofNArgs()) {
      k = op->args_as_KofNArgs()->k();
    }
    const auto code = opcode_to_nombrilo(opcode, k, &negate);
    const auto g = emit_gate(code, *kids);
    lit_of[t] = make_lit(g, !negate);
    if (op->name() && op->name()->size()) {
      add_name(op->name()->str(), make_lit(g, !negate));
    }
  }

  out.offsets.assign(out.operators.size() + 1, 0);
  std::uint32_t acc = 0;
  for (std::size_t i = 0; i < gate_n.size(); ++i) {
    out.offsets[i] = acc;
    acc += gate_n[i];
  }
  out.offsets.back() = acc;

  out.metadata["format"] = kNombriloFormat;
  out.metadata["version"] = kNombriloVersion;
  out.metadata["base_count"] = std::to_string(out.base_count);

  if (options.extras) {
    std::vector<QuerySpec> qs;
    if (sc->queries()) {
      for (const auto *q : *sc->queries()) {
        if (!q) {
          continue;
        }
        QuerySpec spec;
        spec.name = q->name() ? q->name()->str() : "";
        spec.kind = "PROBABILITY";
        if (q->target() < lit_of.size() && lit_of[q->target()]) {
          spec.lit = *lit_of[q->target()];
        } else {
          spec.lit = make_lit(0, true);
        }
        if (q->confidence()) {
          for (auto c : *q->confidence()) {
            spec.confidence.push_back(c);
          }
        }
        qs.push_back(std::move(spec));
      }
    }
    out.metadata["fuzzybits.pcfb.queries"] = dump_queries_json(qs);
    out.metadata["fuzzybits.pcfb.event_ids"] = json_u32_array(sample_event);
    out.metadata["fuzzybits.pcfb.sources"] = dump_sources_json(sample_source);
    out.metadata["fuzzybits.pcfb.metadata"] = dump_pcfb_metadata_json(circuit);
  }

  if (auto r = validate_circuit(out); !r) {
    return std::unexpected(r.error());
  }
  return out;
}

}  // namespace fuzzybits::compile
