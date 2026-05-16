#pragma once

#include "api/ir.h"
#include "api/result.h"

namespace fuzzybits::ir {

[[nodiscard]] inline auto module_descriptor() -> api::ir_backend_descriptor {
    return api::ir_backend_descriptor{
        .provider = api::provider_id("scram"),
        .backend = api::backend_id("ir"),
        .storage_type = api::type_id("ir.module"),
        .kind = api::ir_kind::unknown,
        .name = "SCRAM IR module",
    };
}

} // namespace fuzzybits::ir