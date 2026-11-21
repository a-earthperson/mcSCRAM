// Instantiates one MC kernel at a bitpack width it does not support. Built only
// by the mc.compile_fail.* tests, which pass iff compilation fails and every
// diagnostic is the expected one. Configure with FZB_BITS (128 | 256) and
// exactly one of FZB_KERNEL_ATLEAST / FZB_KERNEL_TALLY.

#include "mc/event/node.h"
#include "mc/event/sample_shape.h"
#include "mc/kernel/gate.h"
#include "mc/kernel/tally.h"

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace {

// Clang caps _BitInt at 128 bits on AArch64, so 256 bits is a limb struct that
// provides every operator the kernels apply to bitpack_t_; any remaining
// diagnostic is then about the width itself.
struct uint256 {
  std::uint64_t limb[4] = {0, 0, 0, 0};

  constexpr uint256() = default;
  constexpr uint256(const unsigned long long v) : limb{v, 0, 0, 0} {}

  constexpr operator std::uint64_t() const { return limb[0]; }

  template <typename I>
  constexpr uint256 operator<<(const I n) const {
    uint256 r;
    const auto s = static_cast<std::size_t>(n);
    for (std::size_t bit = 0; bit + s < 256; ++bit) {
      if ((limb[bit / 64] >> (bit % 64)) & 1u) {
        const std::size_t d = bit + s;
        r.limb[d / 64] |= std::uint64_t{1} << (d % 64);
      }
    }
    return r;
  }

  template <typename I>
  constexpr uint256 operator>>(const I n) const {
    uint256 r;
    const auto s = static_cast<std::size_t>(n);
    for (std::size_t bit = s; bit < 256; ++bit) {
      if ((limb[bit / 64] >> (bit % 64)) & 1u) {
        const std::size_t d = bit - s;
        r.limb[d / 64] |= std::uint64_t{1} << (d % 64);
      }
    }
    return r;
  }

  constexpr uint256 operator&(const uint256 &o) const {
    uint256 r;
    for (std::size_t i = 0; i < 4; ++i) r.limb[i] = limb[i] & o.limb[i];
    return r;
  }

  constexpr uint256 operator|(const uint256 &o) const {
    uint256 r;
    for (std::size_t i = 0; i < 4; ++i) r.limb[i] = limb[i] | o.limb[i];
    return r;
  }

  constexpr uint256 operator~() const {
    uint256 r;
    for (std::size_t i = 0; i < 4; ++i) r.limb[i] = ~limb[i];
    return r;
  }
};

}  // namespace

#if FZB_BITS == 128
using bitpack_t = unsigned __int128;
#elif FZB_BITS == 256
using bitpack_t = uint256;
#else
#error "FZB_BITS must be 128 or 256"
#endif

static_assert(sizeof(bitpack_t) * 8 == FZB_BITS);

using size_t_ = std::size_t;

void instantiate(sycl::queue &queue) {
  const scram::mc::event::sample_shape<size_t_> shape{1, 1};
#if defined(FZB_KERNEL_ATLEAST)
  using kernel_t = scram::mc::kernel::op<scram::core::Connective::kAtleast, bitpack_t, size_t_>;
  const scram::mc::event::atleast_gate_block<bitpack_t, size_t_> block{};
  const kernel_t kernel(block, shape);
  const auto nd = kernel_t::get_range(1, sycl::range<3>{1, 1, 1}, shape);
  queue.submit([&](sycl::handler &cgh) { cgh.parallel_for(nd, kernel); });
#elif defined(FZB_KERNEL_TALLY)
  using kernel_t = scram::mc::kernel::tally<double, bitpack_t, size_t_>;
  const scram::mc::event::tally_block<bitpack_t> block{};
  const kernel_t kernel(block, shape);
  const auto nd = kernel_t::get_range(1, sycl::range<3>{1, 1, 1}, shape);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(nd, [=](sycl::nd_item<3> item) { kernel(item, 1); });
  });
#else
#error "define FZB_KERNEL_ATLEAST or FZB_KERNEL_TALLY"
#endif
}
