/*
 * Copyright (C) 2025 Arjun Earthperson
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "canopy/node.h"
#include "canopy/queue/queueable.h"

#include "pdag.h"

#include <sycl/sycl.hpp>

namespace scram::canopy::queue {

template <typename bitpack_t_ = std::uint64_t, typename prob_t_ = std::double_t, typename size_t_ = std::uint32_t>
class layer_manager {

    using index_t_ = std::int32_t;

    sycl::queue queue_;
    sample_shape<size_t_> sample_shape_;

    std::vector<std::shared_ptr<core::Node>> pdag_nodes_;
    std::unordered_map<index_t_, std::shared_ptr<core::Node>> pdag_nodes_by_index_;
    std::vector<std::vector<std::shared_ptr<core::Node>>> pdag_nodes_by_layer_;

    std::unordered_map<index_t_, std::shared_ptr<queueable_base>> queueables_by_index_;
    std::vector<std::shared_ptr<queueable_base>> queueables_;

    std::unordered_map<index_t_, std::shared_ptr<queueable_base>> tally_queueables_by_index_;

    std::unordered_map<index_t_, tally_event<bitpack_t_> *> allocated_tally_events_by_index_;
    std::unordered_map<index_t_, basic_event<prob_t_, bitpack_t_> *> allocated_basic_events_by_index_;
    std::unordered_map<index_t_, gate<bitpack_t_, size_t_> *> allocated_gates_by_index_;

    std::unordered_map<index_t_, size_t_> accumulated_counts_by_index_;

    static void gather_all_nodes(const std::shared_ptr<core::Gate> &gate,
                                 std::vector<std::shared_ptr<core::Node>> &nodes,
                                 std::unordered_map<std::int32_t, std::shared_ptr<core::Node>> &nodes_by_index);

    static void layered_toposort(core::Pdag *pdag, std::vector<std::shared_ptr<core::Node>> &nodes,
                                 std::unordered_map<index_t_, std::shared_ptr<core::Node>> &nodes_by_index,
                                 std::vector<std::vector<std::shared_ptr<core::Node>>> &nodes_by_layer);

    static void gather_layer_nodes(
        const std::vector<std::shared_ptr<core::Node>> &layer_nodes,
        std::vector<std::shared_ptr<core::Variable>> &out_variables,
        std::unordered_map<core::Connective, std::vector<std::shared_ptr<core::Gate>>> &out_gates_by_type);

    void build_kernels_for_layer(const std::vector<std::shared_ptr<core::Node>> &layer_nodes);

    void map_nodes_by_layer(const std::vector<std::vector<std::shared_ptr<core::Node>>> &nodes_by_layer);

    void fetch_all_tallies();

  public:
    layer_manager(core::Pdag *pdag, size_t_ batch_size, size_t_ bitpacks_per_batch);

    sycl::queue submit_all();

    tally_event<bitpack_t_> tally(index_t_ evt_idx, std::size_t count);

    ~layer_manager();
};
} // namespace scram::canopy::queue

#include "layer_manager.tpp"
