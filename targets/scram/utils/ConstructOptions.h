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
    // clang-format off
        desc.add_options()
            ("help", "Display this help message")
            ("version", "Display version information")
            ("project", OPT_VALUE(path), "Project file with analysis configurations")
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
            ("seed", OPT_VALUE(int), "Seed for the pseudo-random number generator")
            ("output,o", OPT_VALUE(path), "Output file for reports")
            ("no-indent", "Omit indentation whitespace in output XML")
            ("verbosity", OPT_VALUE(int), "Set log verbosity");

        // ------------------------------------------------------------------
        //  Monte-Carlo specific options
        // ------------------------------------------------------------------
        po::options_description mc("Monte Carlo Options");
        mc.add_options()
            ("monte-carlo", "Use the monte-carlo sampling approximation")
            ("num-trials", OPT_VALUE(std::double_t),"Number of Bernoulli trials [0]: Auto")
            ("early-stop", "Stop on convergence, implied if --num-trials unset or 0")
            // ("batch-size", OPT_VALUE(std::size_t),"Batch size (work-group Y dimension)")
            // ("sample-size", OPT_VALUE(std::size_t),"Sample size (work-group Z dimension)")
            ("ci-confidence", OPT_VALUE(double),"Two-sided confidence level used for error estimation")
            ("ci-epsilon", OPT_VALUE(double),"Target margin of error (half-width) for error estimation and early stop")
            ("ci-rel-epsilon", OPT_VALUE(double),"Relative margin of error δ (fraction of p̂). ε = δ*p̂ during run")
            ("ci-pilot", OPT_VALUE(int),"Number of free pilot iterations before convergence checks [3]")
            ("true-prob", OPT_VALUE(double),"Ground truth probability for diagnostics");


        // ------------------------------------------------------------------
        //  Debug options
        // ------------------------------------------------------------------
        po::options_description debug("Debug Options");
        debug.add_options()
            ("serialize", "Serialize the input model without further analysis")
            ("preprocessor", "Stop analysis after the preprocessing step")
            ("print", "Print analysis results in a terminal friendly way")
            ("no-report", "Don't generate analysis report");
        desc.add(mc).add(debug);
    // clang-format on
    return desc;
}
#undef OPT_VALUE
} // namespace ScramCLI