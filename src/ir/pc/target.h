#pragma once

#include "ir/pc/id.h"

#include <cstdint>

namespace fuzzybits::ir::pc {

// Targets are observable nodes in the PC, not solver requests. Result kinds,
// convergence policy, and domain projection payloads live outside this table.
// Classifies why a Boolean node is exposed as an observable target. This is PC
// metadata only: it does not imply graph ownership, a unique root, or a
// fault-tree/event-tree domain object.
enum class observable_role : std::uint8_t {
    // A primary observable requested by the imported model or API request.
    // Multiple primary targets may coexist, and none is privileged as "the"
    // circuit root.
    primary,
    // An auxiliary observable retained for inspection, convergence tracking, or
    // backend diagnostics.
    watch,
    // A stable anchor used to map PC results back to external projection data.
    projection_anchor,
};

// Target ids are stable within this finalized circuit; node ids are storage ids
// and may be remapped by transforms that return provenance maps.
struct target_view final {
    target_id id;
    bool_node_id node;
    observable_role role = observable_role::primary;
};

} // namespace fuzzybits::ir::pc
