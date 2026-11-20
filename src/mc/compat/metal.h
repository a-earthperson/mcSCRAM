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
 * `compile_if_else` usage
 * -----------------------
 * Match AdaptiveCpp `memory_streaming.hpp`: void lambdas that assign into an
 * outer variable. Do not use the documented T() return value; on Metal it
 * lowered, but the returned integer was 0.
 *
 * Removal
 * -------
 * Delete this header and the call sites in `kernel/tally.h` /
 * `queue/layer_manager.tpp` when all of:
 *   1. LLVMToMetal accepts the native 64-bit `sycl::reduce_over_group` on
 *      `std::size_t` (canary `mc.metal_canary.reduce64` xfail-strict flips).
 *   2. LLVMToMetal accepts `double` in device IR, or AdaptiveCpp provides
 *      soft-double (canary `mc.metal_canary.double` flips). Then restore
 *      named `double` fields on `event::tally` if desired.
 *   3. After `queue.wait()`, device `tally.num_one_bits` from a kernel
 *      popcount is host-visible and is not clobbered by a later wait
 *      (canary `mc.metal_canary.device_tally_hits` flips). Then delete
 *      `host_accumulate_if_metal` / `host_publish_if_metal`.
 *
 * 64-bit atomics are **not** shimmed. `mc.metal_canary.atomic64` passed on
 * the hosted macos-26 paravirtual device (eight groups, host saw 8).
 *
 * Metal tally accumulation
 * ------------------------
 * Device popcount of nested USM `tally.buffer` stays 0 on Metal (`n` from a
 * plain `total_bits` store is host-visible; isolated 64-bit `fetch_add`
 * is host-visible). `bitpack_identity` sees bits only via `queue.memcpy`
 * of the producer allocation after wait, not via a host load of USM.
 *
 * Writing hits into the USM `tally` struct is not enough: `wait_sycl_queue`
 * after `single_pass` copies the device image (hits=0) over the host write
 * (run 44642209903: CLI `p_hat=0` with `n` set). Hits therefore live in a
 * host-only map keyed by the tally pointer; `host_publish_if_metal` copies
 * them into the USM struct after the last wait, immediately before collect.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <vector>
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

/// Host-only floating-point overlay of `event::tally` (mean, std_err, ci[4]).
/// Stored as six `uint64_t` so the device struct type contains no `double`.
/// LLVMToMetal walks every identified struct (`Emitter.cpp:emitTypes`) and
/// rejects `Type::getDoubleTy`.
inline constexpr std::size_t kTallyHostFpWords = 6;
inline constexpr std::size_t kTallyHostFpBytes = kTallyHostFpWords * sizeof(std::uint64_t);

template <typename bitpack_t_>
inline constexpr bool group_sum_fits_u32_v =
    (sizeof(bitpack_t_) * 8 * kMaxTallyLocalSize) <= std::numeric_limits<std::uint32_t>::max();

inline bool is_metal_queue(const sycl::queue &queue) {
    return static_cast<int>(queue.get_device().get_backend()) == kMetalRuntimeBackendId;
}

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

inline std::unordered_map<const void *, std::size_t> &metal_host_hits() {
    static thread_local std::unordered_map<const void *, std::size_t> hits;
    return hits;
}

inline std::unordered_map<const void *, const void *> &metal_tally_buffers() {
    static thread_local std::unordered_map<const void *, const void *> buffers;
    return buffers;
}

} // namespace detail

/// Intra-group popcount reduction. Metal path: uint32 then widen. Native path:
/// `reduce_over_group` on `std::size_t` byte-for-byte with pre-shim code.
template <typename Group>
[[gnu::always_inline]] [[gnu::flatten]] inline std::size_t group_sum(const Group &group,
                                                                     const std::size_t local_sum) {
    std::size_t result = 0;
    __acpp_if_target_sscp(namespace jit = sycl::AdaptiveCpp_jit;
                          jit::compile_if_else(
                              jit::reflect<jit::reflection_query::runtime_backend>() == kMetalRuntimeBackendId,
                              [&]() { result = detail::group_sum_metal(group, local_sum); },
                              [&]() { result = detail::group_sum_native(group, local_sum); });
                          return result;);
    return detail::group_sum_native(group, local_sum);
}

/// Metal simd_sum requires uniform control flow. Native path keeps the
/// in-bounds early-return used before this header existed.
[[gnu::always_inline]] [[gnu::flatten]] inline bool join_group_sum(const bool in_bounds) {
    bool join = in_bounds;
    __acpp_if_target_sscp(namespace jit = sycl::AdaptiveCpp_jit;
                          jit::compile_if_else(
                              jit::reflect<jit::reflection_query::runtime_backend>() == kMetalRuntimeBackendId,
                              [&]() { join = true; }, [&]() { join = in_bounds; });
                          return join;);
    return in_bounds;
}

/// Drain the queue on Metal only. AdaptiveCpp Metal `depends_on` does not
/// make prior USM stores visible to the next kernel. No-op on every other
/// backend (host compares `get_backend()` to 5).
[[gnu::always_inline]] inline void wait_if_metal(sycl::queue &queue) {
    if (is_metal_queue(queue)) {
        queue.wait();
    }
}

/// Record the host-tracked producer pointer at tally allocation, before any
/// kernel can rewrite the USM `tally.buffer` field.
template <typename Tally, typename Bitpack>
inline void host_note_tally_buffer(const sycl::queue &queue, Tally *tally, Bitpack *buffer) {
    if (!is_metal_queue(queue) || tally == nullptr) {
        return;
    }
    detail::metal_tally_buffers()[tally] = buffer;
}

/// After a Metal pass, memcpy the producer bitpack allocation and add the
/// popcount into a host-only map. Direct host loads of Metal USM see zeros
/// (`bitpack_identity` uses `queue.memcpy`). Does not write the USM tally:
/// a later `wait()` copies the device image (hits=0) over that store.
template <typename Tally, typename Shape>
inline void host_accumulate_if_metal(sycl::queue &queue, Tally *tally, const Shape &shape) {
    if (!is_metal_queue(queue) || tally == nullptr) {
        return;
    }
    const auto buf_it = detail::metal_tally_buffers().find(tally);
    if (buf_it == detail::metal_tally_buffers().end() || buf_it->second == nullptr) {
        return;
    }
    const std::size_t npacks =
        static_cast<std::size_t>(shape.batch_size) * static_cast<std::size_t>(shape.bitpacks_per_batch);
    if (npacks == 0) {
        return;
    }
    using bitpack_t = std::remove_pointer_t<decltype(Tally::buffer)>;
    const auto *src = static_cast<const bitpack_t *>(buf_it->second);
    std::vector<bitpack_t> host(npacks);
    queue.memcpy(host.data(), src, npacks * sizeof(bitpack_t)).wait();
    std::size_t bits = 0;
    for (std::size_t j = 0; j < npacks; ++j) {
        bits += static_cast<std::size_t>(sycl::popcount(host[j]));
    }
    detail::metal_host_hits()[tally] += bits;
}

/// Copy host-only Metal hits into the USM tally after the last wait, so
/// `collect_tallies` / `fetch_tally_for_event_with_index` see them.
template <typename Tally>
inline void host_publish_if_metal(sycl::queue &queue, Tally *tally) {
    if (!is_metal_queue(queue) || tally == nullptr) {
        return;
    }
    const auto it = detail::metal_host_hits().find(tally);
    if (it != detail::metal_host_hits().end()) {
        tally->num_one_bits = it->second;
    }
}

template <typename Tally>
inline void host_clear_if_metal(Tally *tally) {
    if (tally != nullptr) {
        detail::metal_host_hits().erase(tally);
        detail::metal_tally_buffers().erase(tally);
    }
}

[[gnu::always_inline]] [[gnu::flatten]] inline void add_hits(std::size_t &num_one_bits, const std::size_t group_sum) {
    auto atomic_one_bits = sycl::atomic_ref<std::size_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(num_one_bits);
    atomic_one_bits.fetch_add(group_sum);
}

/// Tally popcount + accumulate. Metal: group leader stores `total_bits` only;
/// hits come from `host_accumulate_if_metal`. Native: original early-return,
/// size_t group reduction, group-leader atomic (pre-shim tally.h).
template <typename Item, typename Tally, typename Shape>
[[gnu::always_inline]] [[gnu::flatten]] inline void count_and_add(const Item &item, Tally *tallies,
                                                                  const std::size_t tally_count,
                                                                  const Shape &sample_shape,
                                                                  const std::uint32_t iteration) {
    using bitpack_t = std::remove_pointer_t<decltype(Tally::buffer)>;
    static constexpr std::size_t num_bits_in_dtype_ = sizeof(bitpack_t) * 8;
    const std::size_t tally_idx = item.get_global_id(0);
    const std::size_t batch_idx = item.get_global_id(1);
    const std::size_t bitpack_id = item.get_global_id(2);

    auto write_total_bits = [&]() {
        const auto total_bits = static_cast<std::size_t>(iteration) * static_cast<std::size_t>(sample_shape.batch_size) *
                                static_cast<std::size_t>(sample_shape.bitpacks_per_batch) * num_bits_in_dtype_;
        tallies[tally_idx].total_bits = total_bits;
    };

    auto native = [&]() {
        if (tally_idx >= tally_count || batch_idx >= sample_shape.batch_size ||
            bitpack_id >= sample_shape.bitpacks_per_batch) {
            return;
        }
        const std::size_t idx = batch_idx * sample_shape.bitpacks_per_batch + bitpack_id;
        const std::size_t local_sum = sycl::popcount(tallies[tally_idx].buffer[idx]);
        const std::size_t group_sum = sycl::reduce_over_group(item.get_group(), local_sum, sycl::plus<>());
        if (item.get_local_linear_id() == 0) {
            write_total_bits();
            add_hits(tallies[tally_idx].num_one_bits, group_sum);
        }
    };

    auto metal = [&]() {
        if (item.get_local_linear_id() != 0) {
            return;
        }
        if (tally_idx >= tally_count) {
            return;
        }
        write_total_bits();
    };

    __acpp_if_target_sscp(namespace jit = sycl::AdaptiveCpp_jit;
                          jit::compile_if_else(
                              jit::reflect<jit::reflection_query::runtime_backend>() == kMetalRuntimeBackendId, metal,
                              native);
                          return;);
    native();
}

} // namespace scram::mc::compat
