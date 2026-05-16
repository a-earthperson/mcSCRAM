#pragma once

#include "ir/pc/id.h"

namespace fuzzybits::ir::pc {

// Edges keep polarity explicit. Packed complement encodings may exist in
// derived views, but the semantic API never relies on signed node ids.
struct edge_ref final {
    bool_node_id source;
    bool complement = false;
};

} // namespace fuzzybits::ir::pc
