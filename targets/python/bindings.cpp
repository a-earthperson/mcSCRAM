#include <libxml/parser.h>
#include <libxml/xmlversion.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "initializer.h"
#include "mc/core/direct_eval.h"
#include "mc/queue/layer_manager.h"
#include "mc/stats/tally_node_map.h"
#include "model.h"
#include "pdag.h"
#include "preprocessor.h"
#include "risk_analysis.h"
#include "settings.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
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
    double overhead_ratio = 0.0;

    py::dict to_graph() const;
    py::dict to_kernel_graph() const;
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

    using bitpack_t = std::uint32_t;
    using size_t_ = std::size_t;
    using prob_t_ = std::double_t;
    using index_t_ = std::int32_t;

    using layer_manager_t = scram::mc::queue::layer_manager<bitpack_t, prob_t_, size_t_>;
    scram::mc::stats::TallyNodeMap tallies;
    if (graph->root_ptr()) {
        tallies.insert_or_assign(
            graph->root_ptr()->index(),
            scram::mc::stats::TallyNode{
                .status = scram::mc::stats::convergence::not_tracked,
                .tally_stats = scram::mc::stats::tally{},
                .node = graph->root_ptr(),
            });
    }

    layer_manager_t manager(
        graph.get(),
        num_trials,
        tallies,
        overhead_ratio);
    const auto kernel_dag = manager.describe_kernel_dag();

    py::dict result = to_graph();
    result["graph_kind"] = "kernel_dag";

    std::unordered_map<index_t_, const layer_manager_t::kernel_dag_node *> kernel_node_by_index;
    for (const auto &node : kernel_dag.nodes) {
        kernel_node_by_index[node.index] = &node;
    }

    py::list graph_nodes = result["nodes"].cast<py::list>();
    for (py::handle item : graph_nodes) {
        py::dict node = py::reinterpret_borrow<py::dict>(item);
        const index_t_ index = node["index"].cast<index_t_>();
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
    sample_shape["total_iterations"] = kernel_dag.total_iterations;

    py::dict kernel;
    kernel["partitions"] = partitions;
    kernel["partition_count"] = kernel_dag.partitions.size();
    kernel["sample_shape"] = sample_shape;
    result["kernel"] = kernel;
    return result;
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
    set_option<double>(options, {"overhead_ratio", "overhead-ratio", "r"}, [&](double value) { settings.overhead_ratio(value); });
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
    compiled->overhead_ratio = settings.overhead_ratio();
    compiled->model = std::move(model);
    compiled->graph = std::move(graph);
    return compiled;
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

} // namespace

PYBIND11_MODULE(mcscram_native, m) {
    m.doc() = "Native Python bindings for the mcSCRAM Monte Carlo probability engine";

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
}
