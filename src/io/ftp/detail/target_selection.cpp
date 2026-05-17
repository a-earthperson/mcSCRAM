#include "io/ftp/detail/target_selection.h"

#include "io/ftp/descriptor.h"

#include <algorithm>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {

    auto selected_targets(ParsedModel const &model, api::option_set const &options) -> std::vector<std::string> {
        if (auto requested = options.get<std::string>(target_option()); requested && !requested->empty()) {
            return {*requested};
        }
        if (!model.explicitTargets.empty()) {
            return model.explicitTargets;
        }

        std::vector<std::string> targets;
        for (auto const &definition : model.definitions) {
            if (!model.referencedNames.contains(definition.name)) {
                targets.push_back(definition.name);
            }
        }
        if (targets.empty() && !model.definitions.empty()) {
            targets.push_back(model.definitions.front().name);
        }
        if (targets.empty() && !model.probabilityNames.empty()) {
            targets = model.probabilityNames;
            std::ranges::sort(targets);
        }
        return targets;
    }

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
