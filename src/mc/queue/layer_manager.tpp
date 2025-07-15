/*
 * Copyright (C) 2025 Arjun Earthperson
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "layer_manager.h"

#include <algorithm>

#include "logger.h"
#include "mc/event/node.h"
#include "mc/working_set.h"
#include "preprocessor.h"

#include "mc/queue/kernel_builder.h"

namespace scram::mc::queue {

template <typename bitpack_t_, typename prob_t_, typename size_t_>
void layer_manager<bitpack_t_, prob_t_, size_t_>::gather_all_nodes(
    const std::shared_ptr<core::Gate> &gate, std::vector<std::shared_ptr<core::Node>> &nodes,
    std::unordered_map<std::int32_t, std::shared_ptr<core::Node>> &nodes_by_index) {
    if (gate->Visited())
        return;
    gate->Visit(1);
    nodes.push_back(gate);
    if (nodes_by_index.contains(gate->index())) {
        LOG(ERROR) << "Found gate with duplicate index while gathering all nodes";
        throw std::runtime_error("gather all nodes failed");
    }
    nodes_by_index[gate->index()] = gate;
    for (const auto &arg : gate->args<core::Gate>()) {
        gather_all_nodes(arg.second, nodes, nodes_by_index);
    }
    for (const auto &arg : gate->args<core::Variable>()) {
        if (!arg.second->Visited()) {
            arg.second->Visit(1);
            nodes.push_back(arg.second);
            if (nodes_by_index.contains(arg.second->index())) {
                LOG(ERROR) << "Found basic event with duplicate index while gathering all nodes";
                throw std::runtime_error("gather all nodes failed");
            }
            nodes_by_index[arg.second->index()] = arg.second;
        }
    }
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
void layer_manager<bitpack_t_, prob_t_, size_t_>::layered_toposort(
    core::Pdag *pdag,
    std::vector<std::shared_ptr<core::Node>> &nodes,
    std::unordered_map<index_t_, std::shared_ptr<core::Node>> &nodes_by_index,
    std::vector<std::vector<std::shared_ptr<core::Node>>> &nodes_by_layer) {
    // Ensure the graph has been topologically sorted, by layer/level
    core::pdag::LayeredTopologicalOrder(pdag);
    // TODO:: Add preprocessing rule for normalizing gates by input count

    // Clear visits for the gathering process
    pdag->Clear<core::Pdag::kVisit>();

    // Collect all nodes
    gather_all_nodes(pdag->root_ptr(), nodes, nodes_by_index);

    // Sort nodes by their order
    std::sort(nodes.begin(), nodes.end(),
              [](const std::shared_ptr<core::Node> &a, const std::shared_ptr<core::Node> &b) {
                  return a->order() < b->order();
              });

    size_t max_layer = nodes.back()->order(); // Since nodes are sorted
    nodes_by_layer.resize(max_layer + 1);

    for (auto &node : nodes) {
        nodes_by_layer[node->order()].push_back(node);
    }

    // For each layer, sort so that variables precede gates, and gates are sorted by their Gate::type()
    for (auto &layer : nodes_by_layer) {
        std::sort(layer.begin(), layer.end(),
                  [](const std::shared_ptr<core::Node> &lhs, const std::shared_ptr<core::Node> &rhs) {
                      // Try casting to Variable
                      auto varL = std::dynamic_pointer_cast<core::Variable>(lhs);
                      auto varR = std::dynamic_pointer_cast<core::Variable>(rhs);

                      // If one is a variable and the other is not, variable goes first
                      if (varL && !varR)
                          return true;
                      if (!varL && varR)
                          return false;

                      // If both are variables, treat them as equivalent in this ordering
                      // (no further ordering required among variables)
                      if (varL && varR)
                          return false;

                      // Otherwise, both must be gates. Compare by gate->type()
                      auto gateL = std::dynamic_pointer_cast<core::Gate>(lhs);
                      auto gateR = std::dynamic_pointer_cast<core::Gate>(rhs);
                      return gateL->type() < gateR->type();
                  });
    }
    LOG(DEBUG5) << "num_nodes: " << nodes.size();
    LOG(DEBUG5) << "num_layers: " << nodes_by_layer.size();
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
void layer_manager<bitpack_t_, prob_t_, size_t_>::gather_layer_nodes(
    const std::vector<std::shared_ptr<core::Node>> &layer_nodes,
    std::vector<std::shared_ptr<core::Variable>> &out_variables,
    std::unordered_map<core::Connective, std::vector<std::shared_ptr<core::Gate>>> &out_gates_by_type) {
    out_variables.clear();
    out_gates_by_type.clear();

    for (auto &node : layer_nodes) {
        // If the node is a Variable, store it
        if (auto var = std::dynamic_pointer_cast<core::Variable>(node)) {
            out_variables.push_back(var);
        }
        // Else if the node is a Gate, group it by Connective type
        else if (auto gate = std::dynamic_pointer_cast<core::Gate>(node)) {
            out_gates_by_type[gate->type()].push_back(gate);
        } else {
            LOG(WARNING) << "gather_layer_nodes: Node " << node->index() << " was neither a Variable nor a Gate.";
        }
    }
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
void layer_manager<bitpack_t_, prob_t_, size_t_>::build_kernels_for_layer(
    const std::vector<std::shared_ptr<core::Node>> &layer_nodes) {
    // Step (1): Partition layer_nodes into Variables and gates_by_type
    std::vector<std::shared_ptr<core::Variable>> variables;
    std::unordered_map<core::Connective, std::vector<std::shared_ptr<core::Gate>>> gates_by_type;
    gather_layer_nodes(layer_nodes, variables, gates_by_type);

    // Step (2): Build a single kernel for all variables in this layer (if any)
    auto be_kernel = build_kernel_for_variables<index_t_, prob_t_, bitpack_t_, size_t_>(
        variables, queue_, sample_shape_, queueables_, queueables_by_index_, allocated_basic_events_by_index_);
    // We could store or log “be_kernel” if we want direct reference, or just rely
    // on the global queueables_ list.

    // Step (3): Build one kernel per gate->type() in this layer
    auto gate_kernels = build_kernels_for_gates<index_t_, prob_t_, bitpack_t_, size_t_>(
        gates_by_type, queue_, sample_shape_, queueables_, queueables_by_index_, allocated_basic_events_by_index_,
        allocated_gates_by_index_);

    // Optionally do something with (be_kernel) and the (gate_kernels) vector.
    // The queueables_ container is updated in each subfunction, so
    // they are already “registered” for execution.
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
void layer_manager<bitpack_t_, prob_t_, size_t_>::map_nodes_by_layer(
    const std::vector<std::vector<std::shared_ptr<core::Node>>> &nodes_by_layer) {
    for (const auto &nodes_in_layer : nodes_by_layer) {
        build_kernels_for_layer(nodes_in_layer);
        // build_tallies_for_layer<index_t_, prob_t_, bitpack_t_, size_t_>(nodes_in_layer, queue_, sample_shape_,
        // queueables_, queueables_by_index_, allocated_basic_events_by_index_, allocated_gates_by_index_,
        // allocated_tally_events_by_index_);
    }
    // last layer gets tallied
    build_tallies_for_layer<index_t_, prob_t_, bitpack_t_, size_t_>(
        nodes_by_layer.back(), queue_, sample_shape_, queueables_, queueables_by_index_,
        allocated_basic_events_by_index_, allocated_gates_by_index_, allocated_tally_events_by_index_);
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
void layer_manager<bitpack_t_, prob_t_, size_t_>::fetch_all_tallies() {
    submit_all().wait_and_throw();
    for (auto &pair : allocated_tally_events_by_index_) {
        const index_t_ index = pair.first;
        const event::tally<bitpack_t_> *tally = pair.second;
        LOG(DEBUG1) << "tally[" << index << "][" << pdag_nodes_by_index_[index].get()->index()
                    << "] :: [std_err] :: [p05, mean, p95] :: " << "[" << tally->std_err << "] :: " << "["
                    << tally->ci[0] << ", " << tally->mean << ", " << tally->ci[1] << "]";
    }
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
layer_manager<bitpack_t_, prob_t_, size_t_>::layer_manager(core::Pdag *pdag, const size_t_ num_trials) {
    // create and sort layers
    layered_toposort(pdag, pdag_nodes_, pdag_nodes_by_index_, pdag_nodes_by_layer_);
    const auto num_nodes = pdag_nodes_.size();
    scheduler_ = scheduler<bitpack_t_>(queue_, num_trials, num_nodes);
    sample_shape_ = scheduler_.SAMPLE_SHAPE;
    
    // Log scheduler configuration
    LOG(DEBUG2) << scheduler_;
    
    map_nodes_by_layer(pdag_nodes_by_layer_);
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
sycl::queue layer_manager<bitpack_t_, prob_t_, size_t_>::submit_all() {
    for (const auto &queueable : queueables_) {
        queueable->submit();
    }
    return queue_;
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
event::tally<bitpack_t_> layer_manager<bitpack_t_, prob_t_, size_t_>::tally(const index_t_ evt_idx) {
    event::tally<bitpack_t_> to_tally;
    if (!allocated_tally_events_by_index_.contains(evt_idx)) {
        LOG(ERROR) << "Unable to tally probability for unknown event with index " << evt_idx;
        return std::move(to_tally);
    }
    LOG(DEBUG1) << "Counting " << scheduler_.TOTAL_ITERATIONS << " tallies for event with index " << evt_idx;

    for (auto i = 0; i < scheduler_.TOTAL_ITERATIONS; i++) {
        fetch_all_tallies();
    }
    const event::tally<bitpack_t_> *computed_tally = allocated_tally_events_by_index_[evt_idx];
    to_tally.num_one_bits = computed_tally->num_one_bits;
    to_tally.mean = computed_tally->mean;
    to_tally.std_err = computed_tally->std_err;
    to_tally.ci = computed_tally->ci;
    return to_tally;
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
layer_manager<bitpack_t_, prob_t_, size_t_>::~layer_manager() {
    // Free allocated basic events
    for (auto &pair : allocated_basic_events_by_index_) {
        event::basic_event<prob_t_, bitpack_t_> *event = pair.second;
        // destroy_basic_event(queue_, event);
    }

    // Free allocated gates
    for (auto &pair : allocated_gates_by_index_) {
        event::gate<bitpack_t_, size_t_> *event = pair.second;
        // destroy_gate(queue_, event);
    }

    // Free allocated tally events
    for (auto &pair : allocated_tally_events_by_index_) {
        event::tally<bitpack_t_> *event = pair.second;
        // destroy_tally_event(queue_, event);
    }
}

} // namespace scram::mc::queue