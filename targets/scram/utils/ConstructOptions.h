/*
 * Copyright (C) 2014-2018 Olzhas Rakhimov
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

/// @file

#pragma once
#include <string>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace ScramCLI {
/// Provides an options value type.
#define OPT_VALUE(type) po::value<type>()->value_name(#type)

/// @returns Command-line option descriptions.
inline po::options_description ConstructOptions() {
    using path = std::string; // To print argument type as path.

        po::options_description desc("Options");
        desc.add_options()
            ("project", OPT_VALUE(path), "project analysis config file")
            ("allow-extern", "**UNSAFE** Allow external libraries")
            ("validate", "Validate input files without analysis")
            ("pdag", "Perform qualitative analysis with PDAG")
            ("bdd", "Perform qualitative analysis with BDD")
            ("zbdd", "Perform qualitative analysis with ZBDD")
            ("mocus", "Perform qualitative analysis with MOCUS")
            ("prime-implicants", "Calculate prime implicants")
            ("probability", "Perform probability analysis")
            ("importance", "Perform importance analysis")
            ("uncertainty", "Perform uncertainty analysis")
            ("ccf", "Perform common-cause failure analysis")
            ("sil", "Compute the Safety Integrity Level metrics")
            ("rare-event", "Use the rare event approximation")
            ("mcub", "Use the MCUB approximation")
            ("limit-order,l", OPT_VALUE(int), "Upper limit for the product order")
            ("cut-off", OPT_VALUE(double), "Cut-off probability for products")
            ("mission-time", OPT_VALUE(double), "System mission time in hours")
            ("time-step", OPT_VALUE(double), "Time step in hours for probability analysis")
            ("num-quantiles", OPT_VALUE(int),"Number of quantiles for distributions")
            ("num-bins", OPT_VALUE(int), "Number of bins for histograms")
            ("output,o", OPT_VALUE(path), "Output file for reports")
            ("no-indent", "Omit indentation whitespace in output XML");

        // ------------------------------------------------------------------
        //  graph compilation specific options
        // ------------------------------------------------------------------
        po::options_description gc("Graph Compilation Options");
        gc.add_options()
            ("pdag", "perform qualitative analysis with PDAG")
            ("bdd", "Perform qualitative analysis with BDD")
            ("zbdd", "Perform qualitative analysis with ZBDD")
            ("mocus", "Perform qualitative analysis with MOCUS");

        // ------------------------------------------------------------------
        //  Monte-Carlo specific options
        // ------------------------------------------------------------------
        po::options_description mc("Monte Carlo Options");
        mc.add_options()
            ("monte-carlo", "enable monte carlo sampling")
            ("early-stop", "stop on convergence (implied if N=0)")
            ("seed", OPT_VALUE(int)->default_value(372), "PRNG seed")
            ("num-trials,N", OPT_VALUE(double)->default_value(0), "# bernoulli trials [N ∈ ℕ₀, 0=auto]")
            ("confidence,a", OPT_VALUE(float)->default_value(0.96), "two-sided conf. lvl [α ∈ (0,1)]")
            ("epsilon,e", OPT_VALUE(double)->default_value(-1.0), "target half-width [ε > 0]")
            ("rel-epsilon,d", OPT_VALUE(double)->default_value(0.01), "relative ε=δ·p̂ [δ > 0]")
            ("burn-in,b", OPT_VALUE(int)->default_value(1<<20), "trials before convergence check [0=off]");

        // ------------------------------------------------------------------
        //  Debug options
        // ------------------------------------------------------------------
        po::options_description debug("Debug Options");
        debug.add_options()
            ("help", "display this help message")
            ("no-report", "don't generate analysis report")
            ("oracle,p", OPT_VALUE(double)->default_value(-1.0), "true p* [p* ∈ [0,∞), -1=off]")
            ("preprocessor", "stop analysis after preprocessing")
            ("print", "print analysis results to terminal")
            ("serialize", "serialize the input model and exit")
            ("verbosity,V", OPT_VALUE(int), "set log verbosity")
            ("version,v", "display version information");
        desc.add(mc).add(debug);
    // clang-format on
    return desc;
}
#undef OPT_VALUE
} // namespace ScramCLI