#pragma once

#include "api/capabilities.h"
#include "api/ids.h"
#include "ir/descriptor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fuzzybits::ir {

struct edge_view final {
    std::uint64_t source = 0;
    bool complement = false;
};

struct node_view final {
    std::uint64_t id = 0;
    std::string kind;
    std::string stable_name;
    std::vector<edge_view> inputs;
};

struct target_view final {
    api::target_id id;
    std::uint64_t node = 0;
};

struct view final {
    ir::descriptor descriptor;
    std::vector<node_view> nodes;
    std::vector<target_view> targets;
    api::capability_set capabilities;
};

} // namespace fuzzybits::ir
