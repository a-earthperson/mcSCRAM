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
/// Constant expressions that cannot have uncertainties.

#pragma once

#include "src/expression.h"

namespace scram::mef {

/// Indicates a constant value.
class ConstantExpression : public Expression {
 public:
  static ConstantExpression kOne;  ///< Constant 1 or True.
  static ConstantExpression kZero;  ///< Constant 0 or False.
  static ConstantExpression kPi;  ///< Constant PI value.

  /// Constructor for constant integer, float, and bool values.
  /// In other words, this constructor is implicitly generic.
  ///
  /// @param[in] value  Numerical value.
  explicit ConstantExpression(double value) : value_(value) {}

  double value()  override { return value_; }
  bool IsDeviate()  override { return false; }

 private:
  double DoSample()  override { return value_; }

  const double value_;  ///< The universal value to represent int, bool, double.
};

}  // namespace scram::mef
