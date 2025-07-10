/*
 * Copyright (C) 2014-2018 Olzhas Rakhimov
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

/// @file
/// SCRAM-specific environment variables.
///
/// All paths are absolute, canonical, and POSIX (with '/' separator).
///
/// @pre The system follows the Filesystem Hierarchy Standard.

#pragma once

#include <string>

namespace scram::env {

/// @returns The embedded RELAX NG schema content for project files.
const std::string_view& project_schema();

/// @returns The embedded RELAX NG schema content for input files.
const std::string_view& input_schema();

/// @returns The embedded RELAX NG schema content for output report files.
const std::string_view& report_schema();

/// @returns The embedded RELAX NG schema content for GUI files.
const std::string_view& gui_schema();

/// @returns The path to the installation directory.
const std::string& install_dir();

}  // namespace scram::env
