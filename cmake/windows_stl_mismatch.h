/* Force-included on WIN32 so LLVM 20 can use a newer MSVC STL.
 * yvals_core.h STL1000 asserts Clang 22 on MSVC 14.52+. */
#define _ALLOW_COMPILER_VERSION_MISMATCH
