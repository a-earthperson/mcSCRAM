#pragma once

#include "api/capabilities.h"
#include "api/ids.h"
#include "api/options.h"

#include <string>

namespace fuzzybits::io {

struct reader_descriptor final {
    api::provider_id provider;
    api::endpoint_id endpoint;
    api::format_id input_format;
    std::string name;
    api::capability_set produces;
    api::option_schema options;
};

struct writer_descriptor final {
    api::provider_id provider;
    api::endpoint_id endpoint;
    api::format_id output_format;
    std::string name;
    api::capability_set accepts;
    api::option_schema options;
};

} // namespace fuzzybits::io
