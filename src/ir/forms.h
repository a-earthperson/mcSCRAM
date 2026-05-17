#pragma once

#include "api/ids.h"

namespace fuzzybits::ir::forms {

[[nodiscard]] inline auto pc() -> api::form_id { return api::form_id("ir.pc"); }
[[nodiscard]] inline auto openpsa() -> api::form_id { return api::form_id("ir.openpsa"); }
[[nodiscard]] inline auto aig() -> api::form_id { return api::form_id("ir.aig"); }
[[nodiscard]] inline auto cnf() -> api::form_id { return api::form_id("ir.cnf"); }
[[nodiscard]] inline auto boolean_dag() -> api::form_id { return api::form_id("ir.boolean_dag"); }
[[nodiscard]] inline auto probabilistic_boolean_dag() -> api::form_id { return api::form_id("ir.probabilistic_boolean_dag"); }

} // namespace fuzzybits::ir::forms
