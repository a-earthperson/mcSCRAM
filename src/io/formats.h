#pragma once

#include "api/ids.h"

namespace fuzzybits::io::formats {

[[nodiscard]] inline auto text() -> api::format_id { return api::format_id("text"); }
[[nodiscard]] inline auto xml() -> api::format_id { return api::format_id("xml"); }
[[nodiscard]] inline auto openpsa_xml() -> api::format_id { return api::format_id("openpsa.xml"); }
[[nodiscard]] inline auto aiger() -> api::format_id { return api::format_id("aiger"); }
[[nodiscard]] inline auto blif() -> api::format_id { return api::format_id("blif"); }
[[nodiscard]] inline auto pcfb_binary() -> api::format_id { return api::format_id("pcfb.binary"); }

} // namespace fuzzybits::io::formats
