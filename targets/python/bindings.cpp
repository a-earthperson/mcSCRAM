#include <libxml/parser.h>
#include <libxml/xmlversion.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sycl/sycl.hpp>

#include "compile/pcfb_writer.h"
#include "compile/sha256.h"
#include "initializer.h"
#include "mc/core/direct_eval.h"
#include "mc/queue/layer_manager.h"
#include "mc/stats/tally_node_map.h"
#include "model.h"
#include "pdag.h"
#include "preprocessor.h"
#include "risk_analysis.h"
#include "runtime/runtime.h"
#include "settings.h"

#include "pcfb.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct ProbabilityResult {
    double p_total = std::numeric_limits<double>::quiet_NaN();
    double analysis_time = 0.0;
};

struct RunSummary {
    double analysis_time = 0.0;
    std::string warnings;
    std::vector<ProbabilityResult> probabilities;
};

struct PdagWatchGuard {
    explicit PdagWatchGuard(const std::unordered_set<const scram::mef::Gate *> *watched_gates) {
        scram::core::Pdag::SetWatchedGates(watched_gates);
    }

    ~PdagWatchGuard() {
        scram::core::Pdag::SetWatchedGates(nullptr);
    }
};

struct CompiledPdag {
    std::vector<std::string> input_files;
    std::unique_ptr<scram::mef::Model> model;
    std::unique_ptr<scram::core::Pdag> graph;
    std::string model_name;
    std::string target_id;
    std::string target_name;
    int compilation_level = 0;
    bool ccf = false;
    bool expand_atleast_gates = false;
    bool expand_xor_gates = false;
    bool recognize_atleast_groups = true;
    bool recognize_xor_groups = true;
    bool keep_null_gates = false;
    std::size_t num_trials = 0;
    std::size_t requested_trials_per_pass = 0;

    py::dict to_graph() const;
    py::dict to_kernel_graph() const;
    py::dict to_execution_plan() const;
};

using PyLayerBitpack = std::uint64_t;
using PyLayerProb = std::double_t;
using PyLayerSize = std::size_t;
using PyLayerIndex = std::int32_t;
using PyLayerManagerNative = scram::mc::queue::layer_manager<PyLayerBitpack, PyLayerProb, PyLayerSize>;

py::dict tally_counter_dict(const scram::mc::event::tally<PyLayerBitpack> &tally) {
    py::dict result;
    result["num_one_bits"] = tally.num_one_bits;
    result["total_bits"] = tally.total_bits;
    return result;
}

void collect_tallyable_nodes(const scram::core::GatePtr &gate,
                             std::unordered_map<PyLayerIndex, std::shared_ptr<scram::core::Node>> &nodes) {
    if (!gate) {
        return;
    }
    if (!nodes.emplace(gate->index(), gate).second) {
        return;
    }

    for (const auto &arg : gate->args<scram::core::Gate>()) {
        collect_tallyable_nodes(arg.second, nodes);
    }

    for (const auto &arg : gate->args<scram::core::Variable>()) {
        nodes.emplace(arg.second->index(), arg.second);
    }
}

scram::mc::stats::TallyNodeMap tally_map_for_indices(const scram::core::Pdag &graph,
                                                     const std::vector<PyLayerIndex> &event_ids) {
    if (!graph.root_ptr()) {
        throw std::runtime_error("cannot create a layer manager for a PDAG without a root");
    }

    std::vector<PyLayerIndex> indices = event_ids;
    if (indices.empty()) {
        indices.push_back(graph.root_ptr()->index());
    }

    std::unordered_map<PyLayerIndex, std::shared_ptr<scram::core::Node>> nodes_by_index;
    collect_tallyable_nodes(graph.root_ptr(), nodes_by_index);

    scram::mc::stats::TallyNodeMap tallies;
    for (const PyLayerIndex index : indices) {
        auto node = nodes_by_index.find(index);
        if (node == nodes_by_index.end()) {
            throw std::invalid_argument("cannot tally unknown PDAG node index: " + std::to_string(index));
        }
        tallies.insert_or_assign(
            index,
            scram::mc::stats::TallyNode{
                .status = scram::mc::stats::convergence::not_tracked,
                .tally_stats = scram::mc::stats::tally{},
                .node = node->second,
            });
    }
    return tallies;
}

py::dict sample_shape_dict(const PyLayerManagerNative &manager) {
    const auto &shape = manager.shaper().SAMPLE_SHAPE;
    py::dict result;
    result["batch_size"] = shape.batch_size;
    result["bitpacks_per_batch"] = shape.bitpacks_per_batch;
    result["bits_per_bitpack"] = sizeof(PyLayerBitpack) * 8;
    result["trials_per_pass"] = shape.batch_size * shape.bitpacks_per_batch * sizeof(PyLayerBitpack) * 8;
    result["total_iterations"] = manager.shaper().TOTAL_ITERATIONS;
    return result;
}

py::dict range3_dict(const scram::mc::queue::range3_description &range) {
    py::dict result;
    result["x"] = range.x;
    result["y"] = range.y;
    result["z"] = range.z;
    result["product"] = range.product();
    return result;
}

py::dict device_dict(const scram::mc::queue::device_description &device) {
    py::dict result;
    result["type"] = device.type;
    result["name"] = device.name;
    result["vendor_id"] = device.vendor_id;
    result["vendor"] = device.vendor;
    result["driver_version"] = device.driver_version;
    result["profile"] = device.profile;
    result["version"] = device.version;
    result["opencl_c_version"] = device.opencl_c_version;
    result["aspects"] = device.aspects;
    result["extensions"] = device.extensions;
    result["max_compute_units"] = device.max_compute_units;
    result["max_clock_frequency"] = device.max_clock_frequency;
    result["max_work_item_dimensions"] = device.max_work_item_dimensions;
    result["max_work_item_sizes_1d"] = range3_dict(device.max_work_item_sizes_1d);
    result["max_work_item_sizes_2d"] = range3_dict(device.max_work_item_sizes_2d);
    result["max_work_item_sizes_3d"] = range3_dict(device.max_work_item_sizes_3d);
    result["work_item_independent_forward_progress"] = device.work_item_independent_forward_progress;
    result["max_work_group_size"] = device.max_work_group_size;
    result["max_num_sub_groups"] = device.max_num_sub_groups;
    result["sub_group_independent_forward_progress"] = device.sub_group_independent_forward_progress;
    result["sub_group_sizes"] = device.sub_group_sizes;
    result["preferred_vector_width_char"] = device.preferred_vector_width_char;
    result["max_mem_alloc_size"] = device.max_mem_alloc_size;
    result["global_mem_size"] = device.global_mem_size;
    result["global_mem_cache_size"] = device.global_mem_cache_size;
    result["global_mem_cache_line_size"] = device.global_mem_cache_line_size;
    result["global_mem_cache_type"] = device.global_mem_cache_type;
    result["local_mem_type"] = device.local_mem_type;
    result["local_mem_size"] = device.local_mem_size;
    return result;
}

py::dict sample_shaper_description_dict(const PyLayerManagerNative::execution_plan_description &plan) {
    const auto &sample = plan.sample_shaper;
    py::dict shape;
    shape["batch_size"] = sample.sample_shape.batch_size;
    shape["bitpacks_per_batch"] = sample.sample_shape.bitpacks_per_batch;
    shape["num_bitpacks"] = sample.sample_shape.num_bitpacks();
    shape["trials_per_pass"] = sample.bits_per_iteration;
    shape["total_iterations"] = sample.total_iterations;

    py::dict resident_memory;
    const auto bytes_per_node = sample.sample_shape.num_bitpacks() * sample.bits_per_bitpack / 8;
    resident_memory["node_count"] = sample.num_nodes;
    resident_memory["bytes_per_node"] = bytes_per_node;
    resident_memory["bytes_total"] = bytes_per_node * sample.num_nodes;
    resident_memory["bits_per_node"] = sample.bits_per_iteration;

    py::dict result;
    result["bitpack_type"] = plan.bitpack_type;
    result["bits_per_bitpack"] = sample.bits_per_bitpack;
    result["requested_trials"] = sample.requested_num_trials;
    result["rounded_trials"] = sample.rounded_num_trials;
    result["total_bits_to_sample"] = sample.total_bits_to_sample;
    result["target_bits_per_iteration"] = sample.target_bits_per_iteration;
    result["bits_per_iteration"] = sample.bits_per_iteration;
    result["total_iterations"] = sample.total_iterations;
    result["requested_trials_per_pass"] = sample.requested_trials_per_pass;
    result["realized_trials_per_pass"] = sample.realized_trials_per_pass;
    result["memory_budget_bitpacks_per_node"] = sample.memory_budget_bitpacks_per_node;
    result["max_device_bytes"] = sample.max_device_bytes;
    result["max_device_bits"] = sample.max_device_bits;
    result["cache_line_bitpacks"] = sample.cache_line_bitpacks;
    result["unlimited_iterations"] = sample.unlimited_iterations;
    result["shape_strategy"] = sample.shape_strategy;
    result["sample_shape"] = shape;
    result["resident_memory"] = resident_memory;
    return result;
}

py::dict partition_dict(const scram::mc::queue::kernel_partition_description &partition) {
    py::dict result;
    result["id"] = partition.id;
    result["layer"] = partition.layer;
    result["kind"] = partition.kind;
    result["connective"] = partition.connective;
    result["operation"] = partition.connective;
    result["node_indices"] = partition.node_indices;
    result["dependency_partition_ids"] = partition.dependency_partition_ids;
    result["node_count"] = partition.node_count;
    result["logical_range"] = range3_dict(partition.logical_range);
    result["local_range"] = range3_dict(partition.local_range);
    result["global_range"] = range3_dict(partition.global_range);
    result["work_group_count"] = partition.work_group_count;
    result["work_item_count"] = partition.work_item_count;
    result["logical_work_item_count"] = partition.logical_work_item_count;
    result["overprovisioned_work_items"] = partition.overprovisioned_work_items;
    result["bitpacks_per_node"] = partition.bitpacks_per_node;
    result["bits_per_node"] = partition.bits_per_node;
    result["bytes_per_node"] = partition.bytes_per_node;
    result["resident_bytes"] = partition.resident_bytes;
    result["min_fan_in"] = partition.min_fan_in;
    result["max_fan_in"] = partition.max_fan_in;
    result["total_fan_in"] = partition.total_fan_in;
    result["positive_input_count"] = partition.positive_input_count;
    result["negated_input_count"] = partition.negated_input_count;
    return result;
}

py::dict dependency_edge_dict(const scram::mc::queue::kernel_dependency_edge &edge) {
    py::dict result;
    result["source_partition"] = edge.source_partition;
    result["target_partition"] = edge.target_partition;
    result["reason"] = edge.reason;
    return result;
}

py::dict execution_node_dict(const scram::mc::queue::execution_node_description &node) {
    py::dict result;
    result["index"] = node.index;
    result["layer"] = node.layer;
    result["partition"] = node.partition;
    result["kind"] = node.kind;
    result["connective"] = node.connective;
    result["fan_in"] = node.fan_in;
    result["positive_input_count"] = node.positive_input_count;
    result["negated_input_count"] = node.negated_input_count;
    result["atleast_min"] = node.atleast_min;
    result["module"] = node.module;
    return result;
}

py::dict execution_plan_dict(const PyLayerManagerNative::execution_plan_description &plan) {
    py::list partitions;
    for (const auto &partition : plan.partitions) {
        partitions.append(partition_dict(partition));
    }

    py::list dependency_edges;
    for (const auto &edge : plan.dependency_edges) {
        dependency_edges.append(dependency_edge_dict(edge));
    }

    py::list nodes;
    for (const auto &node : plan.nodes) {
        nodes.append(execution_node_dict(node));
    }

    py::dict result;
    result["schema"] = plan.schema;
    result["bitpack_type"] = plan.bitpack_type;
    result["bits_per_bitpack"] = plan.bits_per_bitpack;
    result["node_count"] = plan.node_count;
    result["device"] = device_dict(plan.device);
    result["sample_shaper"] = sample_shaper_description_dict(plan);
    result["partitions"] = partitions;
    result["partition_count"] = plan.partitions.size();
    result["dependency_edges"] = dependency_edges;
    result["nodes"] = nodes;
    return result;
}

struct PyLayerManager {
    PyLayerManager(scram::core::Pdag *graph, std::size_t num_trials,
                   const std::vector<PyLayerIndex> &event_ids, std::size_t requested_trials_per_pass) {
        if (!graph) {
            throw std::runtime_error("cannot create a layer manager for an empty compiled PDAG");
        }
        root_index = graph->root_ptr() ? graph->root_ptr()->index() : 0;
        auto tallies = tally_map_for_indices(*graph, event_ids);
        for (const auto &entry : tallies) {
            tally_indices.insert(entry.first);
        }
        manager = std::make_unique<PyLayerManagerNative>(graph, num_trials, tallies, requested_trials_per_pass);
    }

    explicit PyLayerManager(const CompiledPdag &pdag, const std::vector<PyLayerIndex> &event_ids = {})
        : PyLayerManager(pdag.graph.get(), pdag.num_trials, event_ids, pdag.requested_trials_per_pass) {}

    py::dict single_pass(std::optional<PyLayerIndex> event_id = std::nullopt) {
        const PyLayerIndex target = event_id.value_or(root_index);
        if (!tally_indices.contains(target)) {
            throw std::invalid_argument("PDAG node index was not registered for tallying: " + std::to_string(target));
        }
        scram::mc::event::tally<PyLayerBitpack> tally;
        {
            py::gil_scoped_release release;
            tally = manager->single_pass_and_tally(target);
        }
        return tally_counter_dict(tally);
    }

    // Run exactly one pass and return the raw cumulative counters for *every*
    // registered tally node, keyed by PDAG node index.  This is the multi-node
    // analogue of single_pass(): the device tallies all registered nodes during
    // the same pass, so host-side convergence/stats can be computed per node
    // (e.g. gradients across nodes) without re-sampling.
    py::dict single_pass_all() {
        scram::mc::stats::TallyNodeMap tallies;
        {
            py::gil_scoped_release release;
            manager->pass_wait_collect(tallies, 1, 0);
        }
        py::dict result;
        for (auto [index, node_tally] : tallies.tallies()) {
            py::dict counters;
            counters["num_one_bits"] = node_tally.num_one_bits;
            counters["total_bits"] = node_tally.total_bits;
            result[py::int_(index)] = counters;
        }
        return result;
    }

    py::dict execution_plan() const {
        return execution_plan_dict(manager->describe_execution_plan());
    }

    PyLayerIndex root_index = 0;
    std::unordered_set<PyLayerIndex> tally_indices;
    std::unique_ptr<PyLayerManagerNative> manager;
};

void initialize_xml_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        LIBXML_TEST_VERSION
        xmlInitParser();
    });
}

const char *connective_name(scram::core::Connective type) {
    switch (type) {
        case scram::core::kAnd:
            return "and";
        case scram::core::kOr:
            return "or";
        case scram::core::kAtleast:
            return "atleast";
        case scram::core::kXor:
            return "xor";
        case scram::core::kNot:
            return "not";
        case scram::core::kNand:
            return "nand";
        case scram::core::kNor:
            return "nor";
        case scram::core::kNull:
            return "null";
    }
    throw std::logic_error("unknown PDAG connective");
}

const scram::mef::BasicEvent *basic_event_for_variable(const scram::core::Pdag &graph, int index) {
    const auto &basic_events = graph.basic_events();
    const int offset = index - scram::core::Pdag::kVariableStartIndex;
    if (offset < 0 || static_cast<std::size_t>(offset) >= basic_events.size()) {
        return nullptr;
    }
    return basic_events[index];
}

py::dict gate_node_dict(const scram::core::Gate &gate) {
    py::dict node;
    node["index"] = gate.index();
    node["kind"] = "gate";
    node["connective"] = connective_name(gate.type());
    node["min"] = gate.type() == scram::core::kAtleast ? py::cast(gate.min_number()) : py::object(py::none());
    node["module"] = gate.module();
    if (const scram::mef::Gate *origin = gate.mef_origin_ptr()) {
        node["id"] = origin->id();
        node["name"] = origin->name();
        node["label"] = origin->label();
        node["full_path"] = std::string(origin->full_path());
    }
    return node;
}

py::dict variable_node_dict(const scram::core::Pdag &graph, const scram::core::Variable &variable) {
    py::dict node;
    node["index"] = variable.index();
    node["kind"] = "basic_event";
    if (const scram::mef::BasicEvent *event = basic_event_for_variable(graph, variable.index())) {
        node["id"] = event->id();
        node["name"] = event->name();
        node["label"] = event->label();
        node["full_path"] = std::string(event->full_path());
    }
    return node;
}

py::dict constant_node_dict(const scram::core::Constant &constant) {
    py::dict node;
    node["index"] = constant.index();
    node["kind"] = "constant";
    node["value"] = constant.value();
    return node;
}

void append_edge(py::list &edges, int source, int signed_target) {
    py::dict edge;
    edge["source"] = source;
    edge["target"] = std::abs(signed_target);
    edge["complement"] = signed_target < 0;
    edges.append(std::move(edge));
}

void collect_pdag_graph(const scram::core::Pdag &graph,
                        const scram::core::GatePtr &gate,
                        py::list &nodes,
                        py::list &edges,
                        std::unordered_set<int> &seen_nodes,
                        std::unordered_set<int> &expanded_gates) {
    if (!gate) {
        return;
    }

    if (seen_nodes.insert(gate->index()).second) {
        nodes.append(gate_node_dict(*gate));
    }
    if (!expanded_gates.insert(gate->index()).second) {
        return;
    }

    for (const auto &arg : gate->args<scram::core::Gate>()) {
        append_edge(edges, gate->index(), arg.first);
        collect_pdag_graph(graph, arg.second, nodes, edges, seen_nodes, expanded_gates);
    }

    for (const auto &arg : gate->args<scram::core::Variable>()) {
        append_edge(edges, gate->index(), arg.first);
        if (seen_nodes.insert(arg.second->index()).second) {
            nodes.append(variable_node_dict(graph, *arg.second));
        }
    }

    if (gate->constant()) {
        const int signed_index = *gate->args().begin();
        append_edge(edges, gate->index(), signed_index);
        if (seen_nodes.insert(std::abs(signed_index)).second) {
            nodes.append(constant_node_dict(*graph.constant()));
        }
    }
}

py::dict CompiledPdag::to_graph() const {
    py::list nodes;
    py::list edges;
    std::unordered_set<int> seen_nodes;
    std::unordered_set<int> expanded_gates;
    collect_pdag_graph(*graph, graph->root_ptr(), nodes, edges, seen_nodes, expanded_gates);

    py::dict compilation;
    compilation["level"] = compilation_level;
    compilation["ccf"] = ccf;
    compilation["expand_atleast_gates"] = expand_atleast_gates;
    compilation["expand_xor_gates"] = expand_xor_gates;
    compilation["recognize_atleast_groups"] = recognize_atleast_groups;
    compilation["recognize_xor_groups"] = recognize_xor_groups;
    compilation["keep_null_gates"] = keep_null_gates;

    py::dict result;
    result["model"] = model_name.empty() ? py::object(py::none()) : py::cast(model_name);
    result["source"] = input_files.empty() ? py::object(py::none()) : py::cast(input_files.front());
    result["input_files"] = input_files;
    result["target"] = target_id;
    result["target_name"] = target_name;
    result["top"] = graph->root_ptr() ? py::cast(graph->root_ptr()->index()) : py::object(py::none());
    result["root_complement"] = graph->complement();
    result["coherent"] = graph->coherent();
    result["normal"] = graph->normal();
    result["compilation"] = compilation;
    result["nodes"] = nodes;
    result["edges"] = edges;
    return result;
}

py::dict CompiledPdag::to_kernel_graph() const {

    PyLayerManagerNative manager(
        graph.get(),
        num_trials,
        tally_map_for_indices(*graph, {}),
        requested_trials_per_pass);
    const auto kernel_dag = manager.describe_kernel_dag();

    py::dict result = to_graph();
    result["graph_kind"] = "kernel_dag";

    std::unordered_map<PyLayerIndex, const PyLayerManagerNative::kernel_dag_node *> kernel_node_by_index;
    for (const auto &node : kernel_dag.nodes) {
        kernel_node_by_index[node.index] = &node;
    }

    py::list graph_nodes = result["nodes"].cast<py::list>();
    for (py::handle item : graph_nodes) {
        py::dict node = py::reinterpret_borrow<py::dict>(item);
        const PyLayerIndex index = node["index"].cast<PyLayerIndex>();
        auto kernel_node = kernel_node_by_index.find(index);
        if (kernel_node == kernel_node_by_index.end()) {
            continue;
        }
        node["layer"] = kernel_node->second->layer;
        node["partition"] = kernel_node->second->partition;
        node["kernel_kind"] = kernel_node->second->kind;
        node["kernel_connective"] = kernel_node->second->connective;
    }

    py::list partitions;
    for (const auto &partition : kernel_dag.partitions) {
        py::dict item;
        item["id"] = partition.id;
        item["layer"] = partition.layer;
        item["kind"] = partition.kind;
        item["connective"] = partition.connective;
        item["node_indices"] = partition.node_indices;
        item["node_count"] = partition.node_indices.size();
        partitions.append(std::move(item));
    }

    py::dict sample_shape;
    sample_shape["batch_size"] = kernel_dag.batch_size;
    sample_shape["bitpacks_per_batch"] = kernel_dag.bitpacks_per_batch;
    sample_shape["bits_per_bitpack"] = sizeof(PyLayerBitpack) * 8;
    sample_shape["trials_per_pass"] = kernel_dag.batch_size * kernel_dag.bitpacks_per_batch * sizeof(PyLayerBitpack) * 8;
    sample_shape["total_iterations"] = kernel_dag.total_iterations;

    py::dict kernel;
    kernel["partitions"] = partitions;
    kernel["partition_count"] = kernel_dag.partitions.size();
    kernel["sample_shape"] = sample_shape;
    result["kernel"] = kernel;
    return result;
}

py::dict CompiledPdag::to_execution_plan() const {
    PyLayerManagerNative manager(
        graph.get(),
        num_trials,
        tally_map_for_indices(*graph, {}),
        requested_trials_per_pass);

    py::dict plan = execution_plan_dict(manager.describe_execution_plan());
    plan["model"] = model_name.empty() ? py::object(py::none()) : py::cast(model_name);
    plan["source"] = input_files.empty() ? py::object(py::none()) : py::cast(input_files.front());
    plan["input_files"] = input_files;
    plan["target"] = target_id;
    plan["target_name"] = target_name;

    py::dict compilation;
    compilation["level"] = compilation_level;
    compilation["ccf"] = ccf;
    compilation["expand_atleast_gates"] = expand_atleast_gates;
    compilation["expand_xor_gates"] = expand_xor_gates;
    compilation["recognize_atleast_groups"] = recognize_atleast_groups;
    compilation["recognize_xor_groups"] = recognize_xor_groups;
    compilation["keep_null_gates"] = keep_null_gates;
    plan["compilation"] = compilation;
    plan["semantic_graph"] = to_graph();
    return plan;
}

std::optional<py::object> get_option(const py::dict &options, std::initializer_list<const char*> keys) {
    for (const char *key : keys) {
        py::str py_key(key);
        if (options.contains(py_key)) {
            return py::reinterpret_borrow<py::object>(options[py_key]);
        }
    }
    return std::nullopt;
}

template <typename T>
void set_option(const py::dict &options, std::initializer_list<const char*> keys, const auto &setter) {
    if (auto value = get_option(options, keys)) {
        setter(value->cast<T>());
    }
}

bool is_bool_field(const std::string &key) {
    return key == "epsilon_converged" ||
           key == "epsilon_log10_converged" ||
           key == "ci95_covered" ||
           key == "ci99_covered";
}

py::object progress_scalar(const std::string &key, const std::string &value) {
    if (key == "phase") {
        return py::str(value);
    }
    if (is_bool_field(key)) {
        return py::bool_(value == "1" || value == "true" || value == "True");
    }

    const auto has_float_marker = value.find_first_of(".eEinNaA") != std::string::npos;
    try {
        if (has_float_marker) {
            return py::float_(std::stod(value));
        }
        if (!value.empty() && value.front() == '-') {
            return py::int_(std::stoll(value));
        }
        return py::int_(std::stoull(value));
    } catch (const std::exception &) {
        return py::str(value);
    }
}

py::dict progress_dict(const scram::core::Settings::ProgressPairs &pairs) {
    py::dict payload;
    for (const auto &[key, value] : pairs) {
        payload[py::str(key)] = progress_scalar(key, value);
    }
    return payload;
}

struct PyProgressCallback {
    explicit PyProgressCallback(py::object callback) : callback(std::move(callback)) {}

    ~PyProgressCallback() {
        if (Py_IsInitialized()) {
            py::gil_scoped_acquire acquire;
            callback = py::none();
        }
    }

    void operator()(const scram::core::Settings::ProgressPairs &pairs) const {
        py::gil_scoped_acquire acquire;
        callback(progress_dict(pairs));
    }

    py::object callback;
};

scram::core::Settings settings_from_options(const py::dict &options, py::object on_progress) {
    scram::core::Settings settings;
    settings.approximation(scram::core::Approximation::kMonteCarlo);
    settings.probability_analysis(true);

    set_option<int>(options, {"seed"}, [&](int value) { settings.seed(value); });
    set_option<double>(options, {"num_trials", "num-trials", "N"}, [&](double value) { settings.num_trials(value); });
    set_option<double>(options, {"requested_trials_per_pass", "requested-trials-per-pass", "trials_per_pass", "trials-per-pass"}, [&](double value) { settings.requested_trials_per_pass(value); });
    set_option<double>(options, {"delta", "d"}, [&](double value) { settings.ci_rel_margin_error(value); });
    set_option<double>(options, {"burn_in", "burn-in", "b"}, [&](double value) { settings.ci_burnin_trials(value); });
    set_option<double>(options, {"confidence", "a"}, [&](double value) { settings.ci_confidence(value); });
    set_option<std::string>(options, {"policy", "P"}, [&](const std::string &value) { settings.ci_policy(value); });
    set_option<double>(options, {"oracle", "p"}, [&](double value) { settings.oracle_p(value); });
    set_option<double>(options, {"mission_time", "mission-time"}, [&](double value) { settings.mission_time(value); });
    set_option<double>(options, {"time_step", "time-step"}, [&](double value) { settings.time_step(value); });
    set_option<int>(options, {"limit_order", "limit-order", "l"}, [&](int value) { settings.limit_order(value); });
    set_option<double>(options, {"cut_off", "cut-off"}, [&](double value) { settings.cut_off(value); });
    set_option<int>(options, {"compilation_level", "compilation-level", "compilation_passes", "compilation-passes", "c"}, [&](int value) { settings.compilation_level(value); });

    set_option<bool>(options, {"early_stop", "early-stop"}, [&](bool value) { settings.early_stop(value); });
    set_option<bool>(options, {"watch", "watch_mode"}, [&](bool value) { settings.watch_mode(value); });
    set_option<bool>(options, {"expand_atleast_gates", "expand-atleast-gates", "expand_kn", "expand-kn", "no_kn", "no-kn"}, [&](bool value) { settings.expand_atleast_gates(value); });
    set_option<bool>(options, {"expand_xor_gates", "expand-xor-gates", "expand_xor", "expand-xor", "no_xor", "no-xor"}, [&](bool value) { settings.expand_xor_gates(value); });
    set_option<bool>(options, {"recognize_atleast_groups", "recognize-atleast-groups", "recognize_atleast", "recognize-atleast", "recognize_kn", "recognize-kn"}, [&](bool value) { settings.recognize_atleast_groups(value); });
    set_option<bool>(options, {"recognize_xor_groups", "recognize-xor-groups", "recognize_xor", "recognize-xor"}, [&](bool value) { settings.recognize_xor_groups(value); });
    set_option<bool>(options, {"keep_null_gates", "keep-null-gates", "keep_null", "keep-null"}, [&](bool value) { settings.keep_null_gates(value); });
    set_option<bool>(options, {"preprocessor"}, [&](bool value) { settings.preprocessor = value; });
    set_option<bool>(options, {"print"}, [&](bool value) { settings.print = value; });
    set_option<bool>(options, {"importance"}, [&](bool value) { settings.importance_analysis(value); });
    set_option<bool>(options, {"uncertainty"}, [&](bool value) { settings.uncertainty_analysis(value); });
    set_option<bool>(options, {"ccf"}, [&](bool value) { settings.ccf_analysis(value); });
    set_option<bool>(options, {"sil"}, [&](bool value) { settings.safety_integrity_levels(value); });

    if (!on_progress.is_none()) {
        auto callback = std::make_shared<PyProgressCallback>(std::move(on_progress));
        settings.progress_callback([callback](const scram::core::Settings::ProgressPairs &pairs) {
            (*callback)(pairs);
        });
    }

    return settings;
}

RunSummary execute_probability(std::vector<std::string> input_files,
                               scram::core::Settings settings,
                               bool allow_extern) {
    initialize_xml_once();

    settings.input_files(input_files);
    std::unique_ptr<scram::mef::Model> model =
        scram::mef::Initializer(input_files, settings, allow_extern).model();
    settings.model(model.get());

    scram::core::RiskAnalysis analysis(model.get(), settings);
    analysis.Analyze();

    RunSummary summary;
    summary.analysis_time = analysis.analysis_time();
    summary.warnings = analysis.warnings();
    for (const auto &result : analysis.results()) {
        if (result.probability_analysis) {
            summary.probabilities.push_back({
                .p_total = result.probability_analysis->p_total(),
                .analysis_time = result.probability_analysis->analysis_time(),
            });
        }
    }
    return summary;
}

const scram::mef::Gate *find_gate(const scram::mef::Model &model, std::string_view target) {
    for (const scram::mef::Gate &gate : model.gates()) {
        if (gate.id() == target || gate.name() == target || gate.full_path() == target) {
            return &gate;
        }
    }
    return nullptr;
}

const scram::mef::Gate *first_top_gate(const scram::mef::Model &model) {
    for (const scram::mef::FaultTree &fault_tree : model.fault_trees()) {
        if (!fault_tree.top_events().empty()) {
            return fault_tree.top_events().front();
        }
    }
    for (const scram::mef::Gate &gate : model.gates()) {
        return &gate;
    }
    return nullptr;
}

std::unique_ptr<CompiledPdag> run_compile(const std::vector<std::string> &input_files,
                                          const py::dict &options) {
    if (input_files.empty()) {
        throw std::invalid_argument("run_compile requires at least one input file");
    }

    bool allow_extern = false;
    if (auto value = get_option(options, {"allow_extern", "allow-extern"})) {
        allow_extern = value->cast<bool>();
    }

    std::optional<std::string> requested_target;
    if (auto value = get_option(options, {"target", "target_gate", "target-gate", "top", "top_event", "top-event", "gate", "gate_id", "gate-id"})) {
        requested_target = value->cast<std::string>();
    }

    initialize_xml_once();

    auto settings = settings_from_options(options, py::none());
    settings.algorithm(scram::core::Algorithm::kDirect);
    settings.probability_analysis(false);
    settings.input_files(input_files);

    std::unique_ptr<scram::mef::Model> model =
        scram::mef::Initializer(input_files, settings, allow_extern).model();
    settings.model(model.get());

    const scram::mef::Gate *target =
        requested_target ? find_gate(*model, *requested_target) : first_top_gate(*model);
    if (!target) {
        throw std::invalid_argument(
            requested_target
                ? "run_compile could not find target gate: " + *requested_target
                : "run_compile could not find any fault-tree top gate");
    }

    std::unordered_set<const scram::mef::Gate *> watched_gates;
    for (const scram::mef::Gate &gate : model->gates()) {
        watched_gates.insert(&gate);
    }

    std::unique_ptr<scram::core::Pdag> graph;
    {
        PdagWatchGuard watch_guard(&watched_gates);
        graph = std::make_unique<scram::core::Pdag>(*target, settings.ccf_analysis(), model.get());
    }

    if (graph->root_ptr() && !graph->root_ptr()->mef_origin_ptr()) {
        graph->root_ptr()->mef_origin_ptr(target);
    }

    scram::core::CustomPreprocessor<scram::mc::DirectEval> preprocessor(graph.get(), settings);
    preprocessor();

    auto compiled = std::make_unique<CompiledPdag>();
    compiled->input_files = input_files;
    compiled->model_name = model->GetOptionalName();
    if (compiled->model_name.empty()) {
        compiled->model_name = std::filesystem::path(input_files.front()).stem().string();
    }
    compiled->target_id = target->id();
    compiled->target_name = target->name();
    compiled->compilation_level = settings.compilation_level();
    compiled->ccf = settings.ccf_analysis();
    compiled->expand_atleast_gates = settings.expand_atleast_gates();
    compiled->expand_xor_gates = settings.expand_xor_gates();
    compiled->recognize_atleast_groups = settings.recognize_atleast_groups();
    compiled->recognize_xor_groups = settings.recognize_xor_groups();
    compiled->keep_null_gates = settings.keep_null_gates();
    compiled->num_trials = settings.num_trials();
    compiled->requested_trials_per_pass = settings.requested_trials_per_pass();
    compiled->model = std::move(model);
    compiled->graph = std::move(graph);
    return compiled;
}

const char *sycl_device_type_name(sycl::info::device_type type) {
    switch (type) {
        case sycl::info::device_type::cpu:
            return "cpu";
        case sycl::info::device_type::gpu:
            return "gpu";
        case sycl::info::device_type::accelerator:
            return "accelerator";
        case sycl::info::device_type::custom:
            return "custom";
        case sycl::info::device_type::automatic:
            return "automatic";
        case sycl::info::device_type::host:
            return "host";
        case sycl::info::device_type::all:
            return "all";
    }
    return "unknown";
}

std::string sycl_backend_name(const sycl::device &dev) {
    const auto type = dev.get_info<sycl::info::device::device_type>();
    const auto vendor = dev.get_info<sycl::info::device::vendor>();
    const auto name = dev.get_info<sycl::info::device::name>();
    const std::string blob = vendor + " " + name;
    const bool nvidia = blob.find("NVIDIA") != std::string::npos || blob.find("Nvidia") != std::string::npos;
    if (type == sycl::info::device_type::gpu && nvidia) {
        return "cuda";
    }
    if (type == sycl::info::device_type::cpu || type == sycl::info::device_type::host) {
        return "omp";
    }
    if (type == sycl::info::device_type::gpu) {
        return "gpu";
    }
    return sycl_device_type_name(type);
}

template <typename Info>
py::object sycl_info_or_none(const sycl::device &dev) {
    try {
        return py::cast(dev.get_info<Info>());
    } catch (const std::exception &) {
        return py::none();
    }
}

py::list list_sycl_devices() {
    py::list devices;
    for (const sycl::device &dev : sycl::device::get_devices()) {
        py::dict item;
        item["type"] = sycl_device_type_name(dev.get_info<sycl::info::device::device_type>());
        item["backend"] = sycl_backend_name(dev);
        item["name"] = dev.get_info<sycl::info::device::name>();
        item["vendor"] = dev.get_info<sycl::info::device::vendor>();
        item["driver_version"] = sycl_info_or_none<sycl::info::device::driver_version>(dev);
        item["version"] = sycl_info_or_none<sycl::info::device::version>(dev);
        item["max_compute_units"] = sycl_info_or_none<sycl::info::device::max_compute_units>(dev);
        item["global_mem_size"] = sycl_info_or_none<sycl::info::device::global_mem_size>(dev);
        item["max_mem_alloc_size"] = sycl_info_or_none<sycl::info::device::max_mem_alloc_size>(dev);
        item["max_work_group_size"] = sycl_info_or_none<sycl::info::device::max_work_group_size>(dev);
        item["max_clock_frequency"] = sycl_info_or_none<sycl::info::device::max_clock_frequency>(dev);
        devices.append(std::move(item));
    }
    return devices;
}

py::dict run_probability(const std::vector<std::string> &input_files,
                         const py::dict &options,
                         py::object on_progress) {
    if (input_files.empty()) {
        throw std::invalid_argument("run_probability requires at least one input file");
    }

    bool allow_extern = false;
    if (auto value = get_option(options, {"allow_extern", "allow-extern"})) {
        allow_extern = value->cast<bool>();
    }
    auto settings = settings_from_options(options, std::move(on_progress));

    RunSummary summary = execute_probability(input_files, std::move(settings), allow_extern);

    py::list probabilities;
    for (const auto &probability : summary.probabilities) {
        py::dict item;
        item["p_total"] = probability.p_total;
        item["analysis_time"] = probability.analysis_time;
        probabilities.append(item);
    }

    py::dict result;
    result["analysis_time"] = summary.analysis_time;
    result["warnings"] = summary.warnings;
    result["probabilities"] = probabilities;
    return result;
}

fuzzybits::compile::Options compile_options_from_py(const py::dict &options) {
    fuzzybits::compile::Options out;
    set_option<int>(options, {"compilation_level", "compilation-level", "compilation_passes",
                              "compilation-passes", "pdag_c", "c"},
                    [&](int value) { out.compilation_level = value; });
    set_option<bool>(options,
                     {"expand_atleast_gates", "expand-atleast-gates", "expand_kn", "expand-kn", "no_kn", "no-kn"},
                     [&](bool value) { out.expand_atleast_gates = value; });
    set_option<bool>(options, {"expand_xor_gates", "expand-xor-gates", "expand_xor", "expand-xor", "no_xor", "no-xor"},
                     [&](bool value) { out.expand_xor_gates = value; });
    set_option<bool>(options,
                     {"recognize_atleast_groups", "recognize-atleast-groups", "recognize_atleast", "recognize-atleast",
                      "recognize_kn", "recognize-kn"},
                     [&](bool value) { out.recognize_atleast_groups = value; });
    set_option<bool>(options,
                     {"recognize_xor_groups", "recognize-xor-groups", "recognize_xor", "recognize-xor"},
                     [&](bool value) { out.recognize_xor_groups = value; });
    set_option<bool>(options, {"keep_null_gates", "keep-null-gates", "keep_null", "keep-null"},
                     [&](bool value) { out.keep_null_gates = value; });
    set_option<bool>(options, {"ccf"}, [&](bool value) { out.ccf = value; });
    set_option<bool>(options, {"allow_extern", "allow-extern"}, [&](bool value) { out.allow_extern = value; });
    set_option<std::string>(options, {"target", "target_gate", "target-gate", "top", "top_event", "top-event"},
                            [&](const std::string &value) { out.target = value; });
    if (auto value = get_option(options, {"watch", "watch_nodes", "watch-nodes"})) {
        py::object obj = *value;
        if (py::isinstance<py::bool_>(obj)) {
            // Boolean `watch` is watch_mode for run_probability; ignore here.
        } else if (py::isinstance<py::str>(obj)) {
            const auto raw = obj.cast<std::string>();
            std::string cur;
            for (char c : raw) {
                if (c == ',' || c == ';' || c == ':') {
                    if (!cur.empty()) {
                        out.watch.push_back(cur);
                        cur.clear();
                    }
                } else if (!std::isspace(static_cast<unsigned char>(c))) {
                    cur.push_back(c);
                }
            }
            if (!cur.empty()) {
                out.watch.push_back(cur);
            }
        } else if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::bytes>(obj)) {
            for (auto item : obj.cast<py::sequence>()) {
                if (py::isinstance<py::bool_>(item)) {
                    continue;
                }
                auto name = py::str(item).cast<std::string>();
                if (!name.empty()) {
                    out.watch.push_back(std::move(name));
                }
            }
        }
    }
    return out;
}

std::vector<std::byte> bytes_from_py(const py::bytes &data) {
    const std::string raw = data;
    std::vector<std::byte> owned(raw.size());
    if (!raw.empty()) {
        std::memcpy(owned.data(), raw.data(), raw.size());
    }
    return owned;
}

py::bytes py_bytes_from(const std::vector<std::byte> &bytes) {
    return py::bytes(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

const char *opcode_name(pcfb::OpCode op) {
    switch (op) {
        case pcfb::OpCode::BITWISE_NO_OP:
            return "NO_OP";
        case pcfb::OpCode::BITWISE_NOT:
            return "NOT";
        case pcfb::OpCode::BITWISE_AND:
            return "AND";
        case pcfb::OpCode::BITWISE_OR:
            return "OR";
        case pcfb::OpCode::BITWISE_XOR:
            return "XOR";
        case pcfb::OpCode::BITWISE_NAND:
            return "NAND";
        case pcfb::OpCode::BITWISE_NOR:
            return "NOR";
        case pcfb::OpCode::BITWISE_XNOR:
            return "XNOR";
        case pcfb::OpCode::BITWISE_ATLEAST:
            return "ATLEAST";
        case pcfb::OpCode::BITWISE_ATMOST:
            return "ATMOST";
        case pcfb::OpCode::BITWISE_EXACTLY:
            return "EXACTLY";
        case pcfb::OpCode::BITWISE_ZEROS:
            return "ZEROS";
        case pcfb::OpCode::BITWISE_ONES:
            return "ONES";
        case pcfb::OpCode::SAMPLE:
            return "SAMPLE";
        default:
            return "RESERVED";
    }
}

const char *sample_source_name(pcfb::SampleSource s) {
    switch (s) {
        case pcfb::SampleSource::BASIC_EVENT:
            return "BASIC_EVENT";
        case pcfb::SampleSource::COLLAPSED_MODULE:
            return "COLLAPSED_MODULE";
        case pcfb::SampleSource::HOUSE_EVENT:
            return "HOUSE_EVENT";
        default:
            return "UNKNOWN";
    }
}

const char *diagnostic_code_name(pcfb::Diagnostic::Code code) {
    using C = pcfb::Diagnostic::Code;
    switch (code) {
        case C::Ok:
            return "Ok";
        case C::BadIdentifier:
            return "BadIdentifier";
        case C::BadMajorVersion:
            return "BadMajorVersion";
        case C::UnknownOpset:
            return "UnknownOpset";
        case C::MissingRequiredMetadata:
            return "MissingRequiredMetadata";
        case C::MissingEmptyBuffer:
            return "MissingEmptyBuffer";
        case C::BufferIndexOutOfRange:
            return "BufferIndexOutOfRange";
        case C::TensorIndexOutOfRange:
            return "TensorIndexOutOfRange";
        case C::NotTopological:
            return "NotTopological";
        case C::MultipleAssignment:
            return "MultipleAssignment";
        case C::BadArity:
            return "BadArity";
        case C::BadArgs:
            return "BadArgs";
        case C::ReservedOpcode:
            return "ReservedOpcode";
        case C::BadSample:
            return "BadSample";
        case C::InconsistentThreshold:
            return "InconsistentThreshold";
        case C::DuplicateEventId:
            return "DuplicateEventId";
        case C::MissingQuery:
            return "MissingQuery";
        case C::ExtraSubcircuit:
            return "ExtraSubcircuit";
        case C::InvalidFile:
            return "InvalidFile";
    }
    return "Unknown";
}

py::list diagnostics_list(const std::vector<pcfb::Diagnostic> &diags) {
    py::list out;
    for (const auto &d : diags) {
        py::dict item;
        item["code"] = diagnostic_code_name(d.code);
        item["location"] = d.location;
        item["message"] = d.message;
        out.append(std::move(item));
    }
    return out;
}

std::string metadata_entry_string(const auto *kv) {
    if (!kv) {
        return {};
    }
    if (const auto *str = kv->value_as_StringValue(); str && str->value()) {
        return str->value()->str();
    }
    if (const auto *i = kv->value_as_IntValue()) {
        return std::to_string(i->value());
    }
    if (const auto *f = kv->value_as_FloatValue()) {
        std::ostringstream os;
        os << f->value();
        return os.str();
    }
    if (const auto *b = kv->value_as_BytesValue(); b && b->value()) {
        std::vector<std::uint8_t> raw(b->value()->begin(), b->value()->end());
        if (raw.size() == 32) {
            std::array<std::uint8_t, 32> digest{};
            std::copy(raw.begin(), raw.end(), digest.begin());
            return fuzzybits::compile::sha256_hex(digest);
        }
        return "<" + std::to_string(raw.size()) + " bytes>";
    }
    return {};
}

py::dict inspect_circuit(const pcfb::ir::Circuit &circuit) {
    py::dict result;
    result["schema_version"] = circuit.schema_version();

    std::size_t n_tensors = 0;
    std::size_t n_ops = 0;
    std::size_t n_queries = 0;
    std::size_t n_samples = 0;
    std::map<std::string, int> opcode_counts;
    std::map<std::string, int> sample_sources;
    py::list query_names;

    if (circuit.subcircuits() && circuit.subcircuits()->size() > 0) {
        const auto *sc = circuit.subcircuits()->Get(0);
        if (sc->tensors()) {
            n_tensors = sc->tensors()->size();
        }
        if (sc->operators()) {
            n_ops = sc->operators()->size();
            for (const auto *op : *sc->operators()) {
                if (!op) {
                    continue;
                }
                ++opcode_counts[opcode_name(op->opcode())];
                if (op->opcode() == pcfb::OpCode::SAMPLE) {
                    ++n_samples;
                    if (const auto *sa = op->args_as_SampleArgs()) {
                        ++sample_sources[sample_source_name(sa->source())];
                    }
                }
            }
        }
        if (sc->queries()) {
            n_queries = sc->queries()->size();
            for (const auto *q : *sc->queries()) {
                if (!q) {
                    continue;
                }
                std::string name = q->name() ? q->name()->str() : "";
                query_names.append(name.empty() ? ("t" + std::to_string(q->target())) : name);
            }
        }
    }

    result["tensors"] = n_tensors;
    result["operators"] = n_ops;
    result["samples"] = n_samples;
    result["queries"] = n_queries;

    py::dict opcodes;
    for (const auto &[name, n] : opcode_counts) {
        opcodes[py::str(name)] = n;
    }
    result["opcodes"] = std::move(opcodes);

    py::dict sources;
    for (const auto &[name, n] : sample_sources) {
        sources[py::str(name)] = n;
    }
    result["sample_source"] = std::move(sources);
    result["query_names"] = std::move(query_names);

    py::dict metadata;
    if (circuit.get() && circuit.get()->metadata()) {
        for (const auto *kv : *circuit.get()->metadata()) {
            if (!kv || !kv->key()) {
                continue;
            }
            metadata[py::str(kv->key()->str())] = metadata_entry_string(kv);
        }
    }
    result["metadata"] = std::move(metadata);
    result["diagnostics"] = diagnostics_list(pcfb::validate(circuit));
    return result;
}

py::bytes compile_pcfb(const std::vector<std::string> &input_files, const py::dict &options) {
    if (input_files.empty()) {
        throw std::invalid_argument("compile requires at least one input file");
    }
    auto compiled = fuzzybits::compile::compile_files(input_files, compile_options_from_py(options));
    if (!compiled) {
        throw std::runtime_error("compile: " + compiled.error().message);
    }
    return py_bytes_from(compiled->bytes);
}

py::dict inspect_pcfb_bytes(const py::bytes &data) {
    auto owned = bytes_from_py(data);
    auto circuit = pcfb::io::from_bytes(owned);
    if (!circuit) {
        throw std::runtime_error("inspect: " + circuit.error().message);
    }
    return inspect_circuit(*circuit);
}

py::dict inspect_pcfb_path(const std::string &path) {
    auto circuit = pcfb::io::read(path);
    if (!circuit) {
        throw std::runtime_error("inspect: " + circuit.error().message);
    }
    return inspect_circuit(*circuit);
}

py::list query_results_to_py(const std::vector<fuzzybits::runtime::QueryResult> &results) {
    py::list out;
    for (const auto &r : results) {
        py::dict item;
        item["name"] = r.name;
        item["estimate"] = r.estimate;
        item["ci_low"] = r.ci_low;
        item["ci_high"] = r.ci_high;
        item["n"] = r.n;
        item["hits"] = r.hits;
        out.append(std::move(item));
    }
    return out;
}

py::list run_pcfb_bytes(const py::bytes &data, std::uint64_t trials, const std::string &backend) {
    if (trials == 0) {
        throw std::invalid_argument("run: trials must be positive");
    }
    auto owned = bytes_from_py(data);
    fuzzybits::runtime::RunOptions opt;
    opt.trials = trials;
    opt.backend = backend;
    std::vector<fuzzybits::runtime::QueryResult> results;
    {
        py::gil_scoped_release release;
        results = fuzzybits::runtime::run_pcfb(std::span<const std::byte>(owned), opt);
    }
    return query_results_to_py(results);
}

py::dict compilation_from_meta(const fuzzybits::runtime::GraphMeta &meta) {
    py::dict compilation;
    if (!meta.has_compilation) {
        compilation["level"] = py::none();
        compilation["ccf"] = false;
        compilation["expand_atleast_gates"] = py::none();
        compilation["expand_xor_gates"] = py::none();
        compilation["recognize_atleast_groups"] = py::none();
        compilation["recognize_xor_groups"] = py::none();
        compilation["keep_null_gates"] = py::none();
        return compilation;
    }
    compilation["level"] = meta.compilation_level;
    compilation["ccf"] = meta.ccf;
    compilation["expand_atleast_gates"] = meta.expand_atleast_gates;
    compilation["expand_xor_gates"] = meta.expand_xor_gates;
    compilation["recognize_atleast_groups"] = meta.recognize_atleast_groups;
    compilation["recognize_xor_groups"] = meta.recognize_xor_groups;
    compilation["keep_null_gates"] = meta.keep_null_gates;
    return compilation;
}

py::dict annotate_kernel_graph(py::dict result, PyLayerManager &layer_manager) {
    const auto kernel_dag = layer_manager.manager->describe_kernel_dag();
    result["graph_kind"] = "kernel_dag";

    std::unordered_map<PyLayerIndex, const PyLayerManagerNative::kernel_dag_node *> kernel_node_by_index;
    for (const auto &node : kernel_dag.nodes) {
        kernel_node_by_index[node.index] = &node;
    }

    py::list graph_nodes = result["nodes"].cast<py::list>();
    for (py::handle item : graph_nodes) {
        py::dict node = py::reinterpret_borrow<py::dict>(item);
        const PyLayerIndex index = node["index"].cast<PyLayerIndex>();
        auto kernel_node = kernel_node_by_index.find(index);
        if (kernel_node == kernel_node_by_index.end()) {
            continue;
        }
        node["layer"] = kernel_node->second->layer;
        node["partition"] = kernel_node->second->partition;
        node["kernel_kind"] = kernel_node->second->kind;
        node["kernel_connective"] = kernel_node->second->connective;
    }

    py::list partitions;
    for (const auto &partition : kernel_dag.partitions) {
        py::dict item;
        item["id"] = partition.id;
        item["layer"] = partition.layer;
        item["kind"] = partition.kind;
        item["connective"] = partition.connective;
        item["node_indices"] = partition.node_indices;
        item["node_count"] = partition.node_indices.size();
        partitions.append(std::move(item));
    }

    py::dict sample_shape;
    sample_shape["batch_size"] = kernel_dag.batch_size;
    sample_shape["bitpacks_per_batch"] = kernel_dag.bitpacks_per_batch;
    sample_shape["bits_per_bitpack"] = sizeof(PyLayerBitpack) * 8;
    sample_shape["trials_per_pass"] = kernel_dag.batch_size * kernel_dag.bitpacks_per_batch * sizeof(PyLayerBitpack) * 8;
    sample_shape["total_iterations"] = kernel_dag.total_iterations;

    py::dict kernel;
    kernel["partitions"] = partitions;
    kernel["partition_count"] = kernel_dag.partitions.size();
    kernel["sample_shape"] = sample_shape;
    result["kernel"] = kernel;
    return result;
}

struct PyPcfbSession {
    std::vector<std::byte> owned_bytes;
    fuzzybits::runtime::ReconstructedPdag reconstructed;
    std::unique_ptr<PyLayerManager> manager;
    std::vector<PyLayerIndex> tally_ids;
    std::size_t n_trials = 0;
    std::size_t requested_tpp = 0;
    std::uint64_t pass_index = 0;
    bool closed = false;
    std::string target_name;
    std::string model_name;

    explicit PyPcfbSession(const py::bytes &data, const std::string &backend,
                           std::optional<std::uint64_t> trials_per_pass,
                           const std::vector<std::string> &watch = {},
                           std::optional<std::uint64_t> num_trials = std::nullopt) {
        owned_bytes = bytes_from_py(data);
        auto circuit = pcfb::io::from_bytes(owned_bytes);
        if (!circuit) {
            throw std::runtime_error("Session: " + circuit.error().message);
        }
        const auto diags = pcfb::validate(*circuit);
        if (!diags.empty()) {
            std::string msg = "Session: pcfb validation failed (" + std::to_string(diags.size()) +
                              " diagnostics)";
            if (!diags.empty()) {
                msg += ": " + diags.front().message;
            }
            throw std::runtime_error(msg);
        }
        fuzzybits::runtime::apply_backend_mask(backend);
        reconstructed = fuzzybits::runtime::pdag_from_circuit(*circuit);
        const auto &meta = reconstructed.meta;
        target_name = meta.target.empty() ? (reconstructed.queries.empty() ? std::string()
                                                                           : reconstructed.queries.front().name)
                                          : meta.target;
        model_name = meta.model_name.empty() ? target_name : meta.model_name;
        n_trials = static_cast<std::size_t>(num_trials.value_or(0));
        requested_tpp = static_cast<std::size_t>(trials_per_pass.value_or(0));
        for (const auto &q : reconstructed.queries) {
            tally_ids.push_back(static_cast<PyLayerIndex>(q.node_index));
        }
        if (!watch.empty()) {
            for (int index : fuzzybits::runtime::Session::resolve_watch(reconstructed, watch)) {
                tally_ids.push_back(static_cast<PyLayerIndex>(index));
            }
        }
        // Layer-manager / SYCL construction is deferred so DAG views
        // (`graph()`) stay host-only. sample_shape / kernel_graph / step
        // still build the manager on first use.
    }

    void ensure_manager() {
        if (closed) {
            throw std::runtime_error("Session: closed");
        }
        if (manager) {
            return;
        }
        if (!reconstructed.graph) {
            throw std::runtime_error("Session: closed");
        }
        manager = std::make_unique<PyLayerManager>(reconstructed.graph.get(), n_trials, tally_ids,
                                                   requested_tpp);
    }

    void close() {
        manager.reset();
        closed = true;
    }

    py::dict sample_shape() {
        ensure_manager();
        return sample_shape_dict(*manager->manager);
    }

    py::dict graph() const {
        if (!reconstructed.graph) {
            throw std::runtime_error("Session: closed");
        }
        const auto &meta = reconstructed.meta;
        py::list nodes;
        py::list edges;
        std::unordered_set<int> seen_nodes;
        std::unordered_set<int> expanded_gates;
        collect_pdag_graph(*reconstructed.graph, reconstructed.graph->root_ptr(), nodes, edges, seen_nodes,
                           expanded_gates);

        py::dict result;
        result["model"] = model_name.empty() ? py::object(py::none()) : py::cast(model_name);
        result["source"] = meta.source.empty() ? py::object(py::none()) : py::cast(meta.source);
        result["input_files"] = meta.input_files;
        result["target"] = target_name;
        result["target_name"] = meta.target_name.empty() ? target_name : meta.target_name;
        result["top"] = reconstructed.graph->root_ptr() ? py::cast(reconstructed.graph->root_ptr()->index())
                                                        : py::object(py::none());
        result["root_complement"] = reconstructed.graph->complement();
        result["coherent"] = reconstructed.graph->coherent();
        result["normal"] = reconstructed.graph->normal();
        result["compilation"] = compilation_from_meta(meta);
        result["graph_kind"] = "compiled_pdag";
        result["nodes"] = nodes;
        result["edges"] = edges;
        py::list queries;
        for (const auto &q : reconstructed.queries) {
            py::dict item;
            item["name"] = q.name;
            item["index"] = q.node_index;
            queries.append(std::move(item));
        }
        result["query_nodes"] = queries;
        return result;
    }

    py::dict kernel_graph() {
        ensure_manager();
        return annotate_kernel_graph(graph(), *manager);
    }

    py::dict execution_plan() {
        ensure_manager();
        const auto &meta = reconstructed.meta;
        py::dict plan = manager->execution_plan();
        plan["model"] = model_name.empty() ? py::object(py::none()) : py::cast(model_name);
        plan["source"] = meta.source.empty() ? py::object(py::none()) : py::cast(meta.source);
        plan["input_files"] = meta.input_files;
        plan["target"] = target_name;
        plan["target_name"] = meta.target_name.empty() ? target_name : meta.target_name;
        plan["compilation"] = compilation_from_meta(meta);
        plan["semantic_graph"] = graph();
        return plan;
    }

    py::dict step() {
        ensure_manager();
        ++pass_index;
        py::dict raw = manager->single_pass_all();
        py::dict nodes;
        for (auto item : raw) {
            const int index = item.first.cast<int>();
            py::dict counter = py::reinterpret_borrow<py::dict>(item.second);
            py::dict counters;
            counters["index"] = index;
            counters["num_one_bits"] = counter["num_one_bits"];
            counters["total_bits"] = counter["total_bits"];
            counters["one_bits"] = counter["num_one_bits"];
            counters["popcount"] = counter["num_one_bits"];
            nodes[py::int_(index)] = std::move(counters);
        }
        py::dict out;
        out["pass_index"] = pass_index;
        out["trials_per_pass"] = sample_shape()["trials_per_pass"];
        out["target_index"] = reconstructed.queries.empty() ? manager->root_index
                                                            : reconstructed.queries.front().node_index;
        out["nodes"] = std::move(nodes);
        return out;
    }
};

} // namespace

#ifndef FUZZYBITS_NATIVE_MODULE
#define FUZZYBITS_NATIVE_MODULE mcscram_native
#endif

PYBIND11_MODULE(FUZZYBITS_NATIVE_MODULE, m) {
    m.doc() = "Native Python bindings for the fuzzybits Monte Carlo probability engine";
#ifdef FUZZYBITS_NATIVE_MODULE_NAME
    m.attr("__variant_module__") = FUZZYBITS_NATIVE_MODULE_NAME;
#endif

    m.def(
        "list_devices",
        &list_sycl_devices,
        R"pbdoc(
List SYCL devices currently visible to AdaptiveCpp.

Call fuzzybits.init() first so ACPP_VISIBILITY_MASK is applied before the
runtime initialises. Each entry is a dict with type, backend, name, vendor,
and memory/compute attributes.
)pbdoc");

    py::class_<PyLayerManager>(m, "LayerManager")
        .def_property_readonly("root_index", [](const PyLayerManager &layer_manager) {
            return layer_manager.root_index;
        })
        .def_property_readonly("tally_indices", [](const PyLayerManager &layer_manager) {
            return layer_manager.tally_indices;
        })
        .def_property_readonly("node_count", [](const PyLayerManager &layer_manager) {
            return layer_manager.manager->node_count();
        })
        .def_property_readonly("sample_shape", [](const PyLayerManager &layer_manager) {
            return sample_shape_dict(*layer_manager.manager);
        })
        .def_property_readonly("execution_plan", [](const PyLayerManager &layer_manager) {
            return layer_manager.execution_plan();
        })
        .def("single_pass", &PyLayerManager::single_pass, py::arg("event_id") = py::none(), R"pbdoc(
Run exactly one native DirectEval layer-manager pass and return raw cumulative tally counters.

Args:
    event_id: Optional PDAG node index to tally. Defaults to the compiled
        PDAG root. The node must have been registered for tallying when this
        layer manager was created.

Returns:
    A dict with `num_one_bits` and `total_bits`.
)pbdoc")
        .def("single_pass_all", &PyLayerManager::single_pass_all, R"pbdoc(
Run exactly one native DirectEval layer-manager pass and return raw cumulative
tally counters for *every* registered tally node.

All registered nodes are tallied during the same device pass, so this is the
multi-node analogue of `single_pass()` and is the preferred way to drive
host-side convergence/statistics over several nodes simultaneously.

Returns:
    A dict mapping PDAG node index -> {`num_one_bits`, `total_bits`}.
)pbdoc");

    py::class_<CompiledPdag>(m, "Pdag")
        .def_property_readonly("input_files", [](const CompiledPdag &pdag) { return pdag.input_files; })
        .def_property_readonly("model", [](const CompiledPdag &pdag) { return pdag.model_name; })
        .def_property_readonly("target", [](const CompiledPdag &pdag) { return pdag.target_id; })
        .def_property_readonly("compilation_level", [](const CompiledPdag &pdag) { return pdag.compilation_level; })
        .def("to_graph", &CompiledPdag::to_graph, R"pbdoc(
Return a Python dict with the compiled PDAG nodes and signed edges.

The returned graph keeps native PDAG node indices. Edges are directed from
gate nodes to argument nodes; `complement=True` means the argument is negated
in the parent gate formula.
)pbdoc")
        .def("to_kernel_graph", &CompiledPdag::to_kernel_graph, R"pbdoc(
Return a Python dict with the kernel execution DAG.

The graph keeps the compiled PDAG's semantic nodes and edges, and annotates
each node with its execution layer and kernel partition as assigned by the
DirectEval layer manager.
)pbdoc")
        .def("to_execution_plan", &CompiledPdag::to_execution_plan, R"pbdoc(
Return a Python dict describing the DirectEval execution plan for this PDAG.

The execution plan is partition-centric: it includes device capabilities,
sample-shaper decisions, kernel launch geometry, partition dependency edges,
semantic node-to-partition mapping, and the semantic graph as an overlay.
)pbdoc")
        .def("create_layer_manager",
             [](const CompiledPdag &pdag, std::optional<std::vector<PyLayerIndex>> event_ids) {
                 return std::make_unique<PyLayerManager>(pdag, event_ids.value_or(std::vector<PyLayerIndex>{}));
             },
             py::arg("event_ids") = py::none(),
             py::keep_alive<0, 1>(),
             R"pbdoc(
Create a DirectEval layer manager for this compiled PDAG.

The returned object owns the native kernels and accumulates device-side tally
counters across calls to `single_pass()`. By default only the root node is
tallied; pass `event_ids` to register additional PDAG node indices. The compiled
PDAG is kept alive for the lifetime of the returned layer manager.
)pbdoc");

    m.def(
        "run_compile",
        &run_compile,
        py::arg("input_files"),
        py::arg("options") = py::dict(),
        R"pbdoc(
Compile MEF XML input files into a DirectEval PDAG.

Args:
    input_files: MEF XML input files.
    options: Mapping of CLI-like options, including `compilation_level` /
        `compilation_passes`, `ccf`, `no_kn`, `no_xor`,
        `recognize_atleast_groups`, `recognize_xor_groups`, and optional
        `target_gate`/`top_event` selector.

Returns:
    A Pdag object that owns the MEF model and compiled native PDAG.
)pbdoc");

    m.def(
        "run_probability",
        &run_probability,
        py::arg("input_files"),
        py::arg("options") = py::dict(),
        py::arg("on_progress") = py::none(),
        R"pbdoc(
Run a Monte Carlo probability analysis in-process.

Args:
    input_files: MEF XML input files.
    options: Mapping of CLI-like options, using either snake_case or CLI names.
    on_progress: Optional callable receiving a dict for each convergence update.

Returns:
    A dict containing analysis timing, warnings, and probability results.
)pbdoc");

    m.def(
        "compile_pcfb",
        &compile_pcfb,
        py::arg("input_files"),
        py::arg("options") = py::dict(),
        R"pbdoc(
Compile OpenPSA/MEF XML into a pcfb 1.1 buffer.

Args:
    input_files: MEF XML paths.
    options: `compilation_level` / `pdag_c`, `no_kn`, `no_xor`, `ccf`, `target`.

Returns:
    bytes of a validated core-1.1 `.pcfb`.
)pbdoc");

    m.def(
        "inspect_pcfb_bytes",
        &inspect_pcfb_bytes,
        py::arg("data"),
        R"pbdoc(
Summarise an in-memory `.pcfb` (same keys as `fzb inspect --json`, plus `diagnostics`).
)pbdoc");

    m.def(
        "inspect_pcfb_path",
        &inspect_pcfb_path,
        py::arg("path"),
        R"pbdoc(
Summarise a `.pcfb` file (same keys as `fzb inspect --json`, plus `diagnostics`).
)pbdoc");

    m.def(
        "run_pcfb",
        &run_pcfb_bytes,
        py::arg("data"),
        py::arg("trials"),
        py::arg("backend") = "",
        R"pbdoc(
One-shot Monte Carlo of an in-memory `.pcfb`. Same numbers as `fzb run`.

Releases the GIL while the kernel runs. `backend` is `cuda`, `omp`, or empty
(use the mask already applied by fuzzybits.init).
)pbdoc");

    py::class_<PyPcfbSession>(m, "PcfbSession")
        .def(py::init<const py::bytes &, const std::string &, std::optional<std::uint64_t>,
                      const std::vector<std::string> &, std::optional<std::uint64_t>>(),
             py::arg("data"), py::arg("backend") = "", py::arg("trials_per_pass") = py::none(),
             py::arg("watch") = std::vector<std::string>{}, py::arg("num_trials") = py::none())
        .def_property_readonly("sample_shape", &PyPcfbSession::sample_shape)
        .def("graph", &PyPcfbSession::graph)
        .def("kernel_graph", &PyPcfbSession::kernel_graph)
        .def("execution_plan", &PyPcfbSession::execution_plan)
        .def("step", &PyPcfbSession::step, R"pbdoc(
Run exactly one device pass and return cumulative tally counters.

Returns a dict with `pass_index` (1-based), `trials_per_pass`, `target_index`,
and `nodes` mapping PDAG index -> `{num_one_bits, total_bits, one_bits, popcount}`.
Releases the GIL during kernel execution.
)pbdoc")
        .def("close", &PyPcfbSession::close)
        .def("__enter__", [](PyPcfbSession &self) -> PyPcfbSession & { return self; })
        .def("__exit__", [](PyPcfbSession &self, py::object, py::object, py::object) {
            self.close();
            return false;
        });
}
