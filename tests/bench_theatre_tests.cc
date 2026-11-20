// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2014-2018 Olzhas Rakhimov
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

#include "risk_analysis_tests.h"

namespace scram::core::test {

// Benchmark Tests for Theatre fault tree from OpenFTA.
TEST_P(RiskAnalysisTest, Theatre) {
  std::string tree_input = "input/Theatre/theatre.xml";
  settings.probability_analysis(true);
  ASSERT_NO_THROW(ProcessInputFiles({tree_input}));
  ASSERT_NO_THROW(analysis->Analyze());
  // input/Theatre/theatre.xml currently has P(Mains)=1, P(Gen)=0, P(Relay)=0.5
  // so P(M ∧ (G ∨ R)) = 0.5 exactly (OpenFTA's 2.07e-3 figures are gone).
  EXPECT_DOUBLE_EQ(0.5, p_total());
  std::set<std::set<std::string>> mcs = {{"Gen_Fail", "Mains_Fail"},
                                         {"Mains_Fail", "Relay_Fail"}};
  EXPECT_EQ(2, products().size());
  EXPECT_EQ(mcs, products());
}

}  // namespace scram::core::test
