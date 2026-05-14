#pragma once

// Convenience umbrella for consumers that intentionally opt into the concrete
// PC IR contract. Generic orchestration should continue to use api::ir_handle.
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/descriptor.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"
#include "ir/pc/target.h"
