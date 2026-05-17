#pragma once

#include "api/ids.h"

#include <array>
#include <span>
#include <string_view>

namespace fuzzybits::ir::forms {

    struct form_info final {
        api::form_id id;
        std::string_view name;
        std::string_view description;
    };

    [[nodiscard]] inline auto bool_dag() -> api::form_id {
        return api::form_id("ir.bool_dag");
    }

    [[nodiscard]] inline auto nnf() -> api::form_id {
        return api::form_id("ir.nnf");
    }

    [[nodiscard]] inline auto xag() -> api::form_id {
        return api::form_id("ir.xag");
    }

    [[nodiscard]] inline auto flat_nnf() -> api::form_id {
        return api::form_id("ir.f_nnf");
    }

    [[nodiscard]] inline auto dnnf() -> api::form_id {
        return api::form_id("ir.dnnf");
    }

    [[nodiscard]] inline auto deterministic_nnf() -> api::form_id {
        return api::form_id("ir.d_nnf");
    }

    [[nodiscard]] inline auto aig() -> api::form_id {
        return api::form_id("ir.aig");
    }

    [[nodiscard]] inline auto rnf() -> api::form_id {
        return api::form_id("ir.rnf");
    }

    [[nodiscard]] inline auto fprm() -> api::form_id {
        return api::form_id("ir.fprm");
    }

    [[nodiscard]] inline auto cnf() -> api::form_id {
        return api::form_id("ir.cnf");
    }

    [[nodiscard]] inline auto dnf() -> api::form_id {
        return api::form_id("ir.dnf");
    }

    [[nodiscard]] inline auto smooth_dnnf() -> api::form_id {
        return api::form_id("ir.s_dnnf");
    }

    [[nodiscard]] inline auto deterministic_dnnf() -> api::form_id {
        return api::form_id("ir.d_dnnf");
    }

    [[nodiscard]] inline auto pi() -> api::form_id {
        return api::form_id("ir.pi");
    }

    [[nodiscard]] inline auto implicant_cover() -> api::form_id {
        return api::form_id("ir.ip_bcf");
    }

    [[nodiscard]] inline auto pprm() -> api::form_id {
        return api::form_id("ir.pprm");
    }

    [[nodiscard]] inline auto epi() -> api::form_id {
        return api::form_id("ir.epi");
    }

    [[nodiscard]] inline auto essential_implicant() -> api::form_id {
        return api::form_id("ir.eip");
    }

    [[nodiscard]] inline auto smooth_deterministic_dnnf() -> api::form_id {
        return api::form_id("ir.sd_dnnf");
    }

    [[nodiscard]] inline auto bdd() -> api::form_id {
        return api::form_id("ir.bdd");
    }

    [[nodiscard]] inline auto free_bdd() -> api::form_id {
        return api::form_id("ir.f_bdd");
    }

    [[nodiscard]] inline auto obdd() -> api::form_id {
        return api::form_id("ir.obdd");
    }

    [[nodiscard]] inline auto sdd() -> api::form_id {
        return api::form_id("ir.sdd");
    }

    [[nodiscard]] inline auto robdd() -> api::form_id {
        return api::form_id("ir.robdd");
    }

    [[nodiscard]] inline auto probabilistic_boolean_dag() -> api::form_id {
        return api::form_id("ir.probabilistic_boolean_dag");
    }

    [[nodiscard]] inline auto known_form_infos() -> std::span<form_info const> {
        static const std::array infos{
            form_info{bool_dag(), "Boolean DAG", "A Boolean directed acyclic graph with Boolean-valued nodes and target roots."},
            form_info{probabilistic_boolean_dag(), "Probabilistic Boolean DAG",
                      "A Boolean DAG with probability-bearing leaves or side tables sufficient for probabilistic analysis."},
            form_info{nnf(), "NNF", "Negation normal form; negation is pushed to leaves in a Boolean representation."},
            form_info{xag(), "XAG", "XOR-AND graph; a Boolean DAG over XOR, AND, constants, inputs, and edge or leaf polarity."},
            form_info{flat_nnf(), "f-NNF", "Flat negation normal form; a flattened NNF variant suitable for CNF/DNF-style products."},
            form_info{dnnf(), "DNNF", "Decomposable negation normal form; conjunction children are defined over disjoint variable sets."},
            form_info{deterministic_nnf(), "d-NNF", "Deterministic negation normal form; disjunction children are mutually exclusive."},
            form_info{aig(), "AIG", "AND-inverter graph; a Boolean DAG represented with AND nodes and inversion."},
            form_info{rnf(), "RNF", "Reed-Muller normal form; XOR-polynomial representation over Boolean variables."},
            form_info{fprm(), "FPRM", "Fixed-polarity Reed-Muller form; RNF with a fixed variable polarity assignment."},
            form_info{cnf(), "CNF", "Conjunctive normal form; a conjunction of clauses."},
            form_info{dnf(), "DNF", "Disjunctive normal form; a disjunction of conjunction terms."},
            form_info{smooth_dnnf(), "s-DNNF", "Smooth DNNF; every disjunction child mentions the same variable set."},
            form_info{deterministic_dnnf(), "d-DNNF", "Deterministic DNNF; DNNF with mutually exclusive disjunction children."},
            form_info{pi(), "PI", "Prime implicants; irredundant implicant terms that cannot be weakened without losing implication."},
            form_info{implicant_cover(), "IP/BCF", "Implicants or Boolean cover form; a cover-style set of product terms."},
            form_info{pprm(), "PPRM", "Positive-polarity Reed-Muller form; FPRM with all variables in positive polarity."},
            form_info{epi(), "EPI", "Essential prime implicants; prime implicants required by every complete cover."},
            form_info{essential_implicant(), "EIP", "Essential implicants; implicants required by every relevant cover."},
            form_info{smooth_deterministic_dnnf(), "sd-DNNF", "Smooth deterministic DNNF; both smooth and deterministic DNNF."},
            form_info{bdd(), "BDD", "Binary decision diagram; a decision-DAG representation over Boolean variables."},
            form_info{free_bdd(), "f-BDD", "Free BDD; a BDD where each variable is tested at most once along any path."},
            form_info{obdd(), "OBDD", "Ordered BDD; a BDD respecting a single variable order on every path."},
            form_info{sdd(), "SDD", "Sentential decision diagram; a canonical decomposable form relative to a vtree."},
            form_info{robdd(), "ROBDD", "Reduced ordered BDD; an OBDD with redundant tests and duplicate subgraphs eliminated."},
        };
        return infos;
    }

    [[nodiscard]] inline auto info_for(api::form_id const &id) -> form_info const * {
        for (form_info const &info : known_form_infos()) {
            if (info.id == id) {
                return &info;
            }
        }
        return nullptr;
    }

} // namespace fuzzybits::ir::forms
