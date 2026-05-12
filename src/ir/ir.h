#pragma once

#include "api/ir.h"
#include "api/result.h"

namespace scram::ir {

[[nodiscard]] inline auto module_descriptor() -> api::ir_backend_descriptor {
  return api::ir_backend_descriptor{
      .backend = api::backend_id("ir"),
      .storage_type = api::type_id("ir.module"),
      .kind = api::ir_kind::unknown,
      .name = "SCRAM IR module",
  };
}

}  // namespace scram::ir