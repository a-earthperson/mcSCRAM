#pragma once

#include "api/ids.h"

#include <memory>
#include <utility>

namespace fuzzybits::api {

class handle final {
  public:
    handle() = default;

    template <typename storage_type>
    [[nodiscard]] static auto make(std::shared_ptr<storage_type const> storage, type_id storage_type_id) -> handle {
        return handle(std::move(storage), std::move(storage_type_id));
    }

    [[nodiscard]] auto valid() const noexcept -> bool { return storage_ != nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
    [[nodiscard]] auto storage_type() const noexcept -> type_id const & { return storage_type_; }
    [[nodiscard]] auto raw_storage() const noexcept -> void const * { return storage_.get(); }

    template <typename storage_type> [[nodiscard]] auto storage_as() const noexcept -> storage_type const * {
        return static_cast<storage_type const *>(storage_.get());
    }

  private:
    handle(std::shared_ptr<void const> storage, type_id storage_type_id)
        : storage_(std::move(storage)), storage_type_(std::move(storage_type_id)) {}

    std::shared_ptr<void const> storage_;
    type_id storage_type_;
};

} // namespace fuzzybits::api
