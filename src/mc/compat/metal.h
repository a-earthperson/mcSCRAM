/**
 * @file metal.h
 * @brief JIT-time Metal accommodations for the Monte Carlo kernels.
 *
 * Narrow, removable shims selected with AdaptiveCpp
 * `sycl::AdaptiveCpp_jit::compile_if_else` on the SSCP generic target.
 * Non-Metal backends (OMP host, CUDA SSCP, CUDA SMCP `cuda:sm_70`, Windows,
 * Linux) take the native path: outside SSCP the reflection builtins do not
 * exist, so the native call is compiled in directly.
 *
 * Metal identifier
 * ----------------
 * `reflection_query::runtime_backend` returns `sycl::backend` cast to `int`.
 * At `ACPP_REF_METAL` (96c7d17a) that is `hipsycl::rt::backend_id`:
 *   cuda=0, hip=1, level_zero=2, ocl=3, omp=4, **metal=5**, vk=6.
 * v25.10.0 has no `metal` enumerator, so this header compares the reflected
 * integer to `kMetalRuntimeBackendId` and compiles on both pins. OMP inside
 * a Metal AdaptiveCpp build reflects `omp` (4) and keeps the native path.
 *
 * Removal
 * -------
 * Delete this header and the call sites in `kernel/tally.h` when all of:
 *   1. LLVMToMetal accepts the native 64-bit `sycl::reduce_over_group` on
 *      `std::size_t` (canary `mc.metal_canary.reduce64` xfail-strict flips).
 *   2. LLVMToMetal accepts `double` in device IR, or AdaptiveCpp provides
 *      soft-double (canary `mc.metal_canary.double` flips). Then restore
 *      named `double` fields on `event::tally` if desired.
 *   3. Native `atomic_ref<std::size_t>::fetch_add` is host-visible on Metal
 *      (canary `mc.metal_canary.atomic64` flips). Then `add_hits` can drop
 *      the uint32 path.
 *
 * Evidence for the 64-bit atomic shim (macos-metal 44629827202, hosted
 * macos-26 paravirtual device): after the double fields left device IR, the
 * tally kernel completed and a plain store of `total_bits` was host-visible
 * (`n=65536`) but `num_one_bits` stayed 0. AdaptiveCpp emulates atomic64 with
 * a hashed lock table and 32-bit half stores; host-device atomics are not
 * guaranteed (install-metal.md). Metal `add_hits` therefore uses a native
 * 32-bit `fetch_add` on the low word. A group sum is ≤ 2^17, and CI trial
 * counts keep the running total well below 2^32.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

namespace scram::mc::compat {

/// `sycl::backend` / `rt::backend_id::metal` at ACPP_REF_METAL. Not an
/// enumerator on v25.10.0 — keep this a plain `int`.
inline constexpr int kMetalRuntimeBackendId = 5;

/// Bound used by `kernel::tally` / `working_set` GPU local ranges (SYCL GPU
/// max work-group size is 1024 on CUDA and Metal). A group popcount is at
/// most `bits(bitpack_t) * local_size` ≤ 128 × 1024 = 2^17.
inline constexpr std::size_t kMaxTallyLocalSize = 1024;
inline constexpr std::size_t kMaxBitpackBits = 128;

static_assert(kMaxBitpackBits * kMaxTallyLocalSize <= std::numeric_limits<std::uint32_t>::max(),
              "Metal uint32 group-sum cannot overflow for the supported pack width");
static_assert(kMaxBitpackBits * kMaxTallyLocalSize <= (std::size_t{1} << 17),
              "group-sum bound is bits(bitpack) x local_size <= 2^17");
static_assert(sizeof(std::size_t) >= sizeof(std::uint32_t),
              "Metal add_hits overlays the low 32 bits of std::size_t");

/// Host-only floating-point overlay of `event::tally` (mean, std_err, ci[4]).
/// Stored as six `uint64_t` so the device struct type contains no `double`.
/// LLVMToMetal walks every identified struct (`Emitter.cpp:emitTypes`) and
/// rejects `Type::getDoubleTy`.
inline constexpr std::size_t kTallyHostFpWords = 6;
inline constexpr std::size_t kTallyHostFpBytes = kTallyHostFpWords * sizeof(std::uint64_t);

template <typename bitpack_t_>
inline constexpr bool group_sum_fits_u32_v =
    (sizeof(bitpack_t_) * 8 * kMaxTallyLocalSize) <= std::numeric_limits<std::uint32_t>::max();

namespace detail {

template <typename Group>
[[gnu::always_inline]] [[gnu::flatten]] inline std::size_t group_sum_native(const Group &group,
                                                                            const std::size_t local_sum) {
    return sycl::reduce_over_group(group, local_sum, sycl::plus<>());
}

template <typename Group>
[[gnu::always_inline]] [[gnu::flatten]] inline std::size_t group_sum_metal(const Group &group,
                                                                           const std::size_t local_sum) {
    const auto local32 = static_cast<std::uint32_t>(local_sum);
    const auto group32 = sycl::reduce_over_group(group, local32, sycl::plus<>());
    return static_cast<std::size_t>(group32);
}

[[gnu::always_inline]] [[gnu::flatten]] inline void add_hits_native(std::size_t &num_one_bits,
                                                                    const std::size_t group_sum) {
    auto atomic_one_bits = sycl::atomic_ref<std::size_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(num_one_bits);
    atomic_one_bits.fetch_add(group_sum);
}

/// Native 32-bit atomic on the low word of `num_one_bits`. Little-endian
/// overlay; wraps at 2^32 (above any `ctest -L mc` trial count).
[[gnu::always_inline]] [[gnu::flatten]] inline void add_hits_metal(std::size_t &num_one_bits,
                                                                   const std::size_t group_sum) {
    auto *low = reinterpret_cast<std::uint32_t *>(&num_one_bits);
    auto atomic_low = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                       sycl::access::address_space::global_space>(*low);
    atomic_low.fetch_add(static_cast<std::uint32_t>(group_sum));
}

} // namespace detail

/// Intra-group popcount reduction. Metal path: uint32 then widen. Native path:
/// `reduce_over_group` on `std::size_t` byte-for-byte with pre-shim code.
template <typename Group>
[[gnu::always_inline]] [[gnu::flatten]] inline std::size_t group_sum(const Group &group,
                                                                     const std::size_t local_sum) {
    std::size_t result = 0;
    __acpp_if_target_sscp(namespace jit = sycl::AdaptiveCpp_jit;
                          result = jit::compile_if_else(
                              jit::reflect<jit::reflection_query::runtime_backend>() == kMetalRuntimeBackendId,
                              [&]() { return detail::group_sum_metal(group, local_sum); },
                              [&]() { return detail::group_sum_native(group, local_sum); });
                          return result;);
    return detail::group_sum_native(group, local_sum);
}

/// Metal simd_sum requires uniform control flow. Native path keeps the
/// in-bounds early-return used before this header existed.
[[gnu::always_inline]] [[gnu::flatten]] inline bool join_group_sum(const bool in_bounds) {
    bool join = in_bounds;
    __acpp_if_target_sscp(namespace jit = sycl::AdaptiveCpp_jit;
                          join = jit::compile_if_else(
                              jit::reflect<jit::reflection_query::runtime_backend>() == kMetalRuntimeBackendId,
                              [&]() { return true; }, [&]() { return in_bounds; });
                          return join;);
    return in_bounds;
}

/// Group-leader hit accumulator. Native `atomic_ref<std::size_t>::fetch_add`.
/// Metal: uint32 `fetch_add` on the low word (see file comment).
[[gnu::always_inline]] [[gnu::flatten]] inline void add_hits(std::size_t &num_one_bits, const std::size_t group_sum) {
    __acpp_if_target_sscp(namespace jit = sycl::AdaptiveCpp_jit;
                          jit::compile_if_else(
                              jit::reflect<jit::reflection_query::runtime_backend>() == kMetalRuntimeBackendId,
                              [&]() { detail::add_hits_metal(num_one_bits, group_sum); },
                              [&]() { detail::add_hits_native(num_one_bits, group_sum); });
                          return;);
    detail::add_hits_native(num_one_bits, group_sum);
}

} // namespace scram::mc::compat
