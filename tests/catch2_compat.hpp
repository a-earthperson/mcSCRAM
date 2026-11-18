#pragma once

// Catch2 v3 compatibility shim for the pre-#31 suite, which included
// <catch2/catch.hpp> (v2 amalgam) and used unqualified Approx.
#include <catch2/catch_all.hpp>

using Catch::Approx;
using Catch::Generators::as;
using Catch::Generators::table;
using Catch::Generators::values;
