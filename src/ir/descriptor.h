#pragma once

#include "api/capabilities.h"
#include "api/ids.h"
#include "api/options.h"

#include <string>
#include <vector>

namespace fuzzybits::ir {

struct descriptor final {
    api::provider_id provider;
    api::endpoint_id endpoint;
    api::type_id storage_type;
    api::form_id form;
    std::string name;
    api::capability_set capabilities;
};

struct conversion_descriptor final {
    api::provider_id provider;
    api::endpoint_id endpoint;
    std::string name;
    api::form_id input_form;
    api::form_id output_form;
    api::capability_set required;
    api::capability_set produced;
    api::option_schema options;
};

} // namespace fuzzybits::ir
