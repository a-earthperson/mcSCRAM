#pragma once

#include "api/ids.h"

namespace fuzzybits::io::formats {

    namespace openpsa {
        [[nodiscard]] inline auto report() -> api::format_id {
            return api::format_id("openpsa.report.xml");
        }
        [[nodiscard]] inline auto project() -> api::format_id {
            return api::format_id("openpsa.project.xml");
        }
        [[nodiscard]] inline auto input() -> api::format_id {
            return api::format_id("openpsa.input.xml");
        }
    }

    [[nodiscard]] inline auto fta() -> api::format_id {
        return api::format_id("fta");
    }

    [[nodiscard]] inline auto aiger() -> api::format_id {
        return api::format_id("aig");
    }
    [[nodiscard]] inline auto blif() -> api::format_id {
        return api::format_id("blif");
    }


    [[nodiscard]] inline auto pcfb() -> api::format_id {
        return api::format_id("pcfb");
    }

} // namespace fuzzybits::io::formats
