#pragma once

#include "ir/handle.h"
#include "ir/pc/descriptor.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"
#include "ir/pc/target.h"

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace fuzzybits::ir::pc {

class builder;

namespace detail {

struct circuit_storage final {
    std::vector<op_code> op_codes;
    std::vector<std::uint32_t> operand_offsets;
    std::vector<std::uint32_t> operand_counts;
    std::vector<edge_ref> edges;
    std::vector<probability_source_id> probability_source_refs;
    std::vector<std::uint32_t> cardinality_low;
    std::vector<std::uint32_t> cardinality_high;
    std::vector<std::uint8_t> constant_values;
    std::vector<probability_source> probability_sources;
    std::vector<target_view> targets;
};

} // namespace detail

class circuit final {
  public:
    circuit() = default;

    [[nodiscard]] auto valid() const noexcept -> bool { return storage_ != nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
    [[nodiscard]] auto descriptor() const noexcept -> ir::descriptor const & { return descriptor_; }
    [[nodiscard]] auto shape() const noexcept -> shape_summary const & { return shape_; }

    [[nodiscard]] auto node_count() const noexcept -> std::uint32_t {
        return storage_ == nullptr ? 0U : static_cast<std::uint32_t>(storage_->op_codes.size());
    }

    [[nodiscard]] auto edge_count() const noexcept -> std::uint32_t {
        return storage_ == nullptr ? 0U : static_cast<std::uint32_t>(storage_->edges.size());
    }

    [[nodiscard]] auto op_codes() const noexcept -> std::span<op_code const> {
        return storage_ == nullptr ? std::span<op_code const>{} : std::span<op_code const>{storage_->op_codes};
    }

    [[nodiscard]] auto operand_offsets() const noexcept -> std::span<std::uint32_t const> {
        return storage_ == nullptr ? std::span<std::uint32_t const>{}
                                   : std::span<std::uint32_t const>{storage_->operand_offsets};
    }

    [[nodiscard]] auto operand_counts() const noexcept -> std::span<std::uint32_t const> {
        return storage_ == nullptr ? std::span<std::uint32_t const>{}
                                   : std::span<std::uint32_t const>{storage_->operand_counts};
    }

    [[nodiscard]] auto edges() const noexcept -> std::span<edge_ref const> {
        return storage_ == nullptr ? std::span<edge_ref const>{} : std::span<edge_ref const>{storage_->edges};
    }

    [[nodiscard]] auto probability_source_refs() const noexcept -> std::span<probability_source_id const> {
        return storage_ == nullptr ? std::span<probability_source_id const>{}
                                   : std::span<probability_source_id const>{storage_->probability_source_refs};
    }

    [[nodiscard]] auto probability_sources() const noexcept -> std::span<probability_source const> {
        return storage_ == nullptr ? std::span<probability_source const>{}
                                   : std::span<probability_source const>{storage_->probability_sources};
    }

    [[nodiscard]] auto cardinality_low() const noexcept -> std::span<std::uint32_t const> {
        return storage_ == nullptr ? std::span<std::uint32_t const>{}
                                   : std::span<std::uint32_t const>{storage_->cardinality_low};
    }

    [[nodiscard]] auto cardinality_high() const noexcept -> std::span<std::uint32_t const> {
        return storage_ == nullptr ? std::span<std::uint32_t const>{}
                                   : std::span<std::uint32_t const>{storage_->cardinality_high};
    }

    [[nodiscard]] auto constant_values() const noexcept -> std::span<std::uint8_t const> {
        return storage_ == nullptr ? std::span<std::uint8_t const>{}
                                   : std::span<std::uint8_t const>{storage_->constant_values};
    }

    [[nodiscard]] auto targets() const noexcept -> std::span<target_view const> {
        return storage_ == nullptr ? std::span<target_view const>{} : std::span<target_view const>{storage_->targets};
    }

    [[nodiscard]] auto operands(bool_node_id node) const noexcept -> std::span<edge_ref const> {
        if (storage_ == nullptr || !node.valid() || node.raw() >= storage_->op_codes.size()) {
            return {};
        }
        const auto offset = storage_->operand_offsets[node.raw()];
        const auto count = storage_->operand_counts[node.raw()];
        return std::span<edge_ref const>{storage_->edges}.subspan(offset, count);
    }

    [[nodiscard]] auto to_handle() const -> ir::handle {
        if (!valid()) {
            return {};
        }
        return ir::handle::make(std::make_shared<circuit const>(*this), descriptor_);
    }

  private:
    friend class builder;

    circuit(std::shared_ptr<detail::circuit_storage const> storage, shape_summary shape)
        : storage_(std::move(storage)), shape_(std::move(shape)), descriptor_(make_descriptor(shape_)) {}

    std::shared_ptr<detail::circuit_storage const> storage_;
    shape_summary shape_;
    ir::descriptor descriptor_ = make_descriptor();
};

[[nodiscard]] inline auto is_pc(ir::handle const &handle) -> bool {
    return handle.valid() && handle.storage_type() == storage_type_id();
}

[[nodiscard]] inline auto circuit_from(ir::handle const &handle) -> circuit const * {
    return is_pc(handle) ? handle.storage_as<circuit>() : nullptr;
}

} // namespace fuzzybits::ir::pc
