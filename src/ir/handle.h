#pragma once

#include "api/handle.h"
#include "ir/descriptor.h"

#include <memory>
#include <utility>

namespace fuzzybits::ir {

class handle final {
  public:
    handle() = default;

    template <typename storage_type>
    [[nodiscard]] static auto make(std::shared_ptr<storage_type const> storage, descriptor descriptor) -> handle {
        const auto storage_type_id = descriptor.storage_type;
        return handle(api::handle::make(std::move(storage), std::move(storage_type_id)), std::move(descriptor));
    }

    [[nodiscard]] auto valid() const noexcept -> bool { return storage_.valid(); }
    [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
    [[nodiscard]] auto descriptor() const noexcept -> ir::descriptor const & { return descriptor_; }
    [[nodiscard]] auto storage_type() const noexcept -> api::type_id const & { return storage_.storage_type(); }

    template <typename storage_type> [[nodiscard]] auto storage_as() const noexcept -> storage_type const * {
        return storage_.storage_as<storage_type>();
    }

  private:
    handle(api::handle storage, ir::descriptor descriptor)
        : storage_(std::move(storage)), descriptor_(std::move(descriptor)) {}

    api::handle storage_;
    ir::descriptor descriptor_;
};

} // namespace fuzzybits::ir
