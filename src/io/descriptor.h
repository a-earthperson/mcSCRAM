#pragma once

#include "api/capabilities.h"
#include "api/ids.h"
#include "api/options.h"
#include "api/result.h"
#include "ir/handle.h"

#include <filesystem>
#include <string>

namespace fuzzybits::io {

    // read
    struct reader_descriptor final {
        std::string name;
        api::provider_id provider;
        api::endpoint_id endpoint;
        api::format_id input_format;
        api::capability_set produces;
        api::option_schema options;
    };
    struct read_request final {
        std::filesystem::path path;
        api::format_id format;
        api::option_set options;
    };
    using read_result = api::result<ir::handle>;

    // write
    struct writer_descriptor final {
        std::string name;
        api::provider_id provider;
        api::endpoint_id endpoint;
        api::format_id output_format;
        api::capability_set accepts;
        api::option_schema options;
    };
    struct write_request final {
        api::format_id format;
        api::option_set options;
    };
    using write_result = api::result<bool>;

} // namespace fuzzybits::io
