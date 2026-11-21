#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "mc/event/node.h"
#include "mc/event/sample_shape.h"
#include "mc/kernel/tally.h"

#include <array>
#include <bit>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

// Widths of 128 and 256 bits are rejected at compile time; see
// bitpack_width_compile_fail.cc and the mc.compile_fail.* tests.

namespace {

using prob_t = double;
using size_t_ = std::size_t;

constexpr std::size_t kBatchSize = 4;
constexpr std::size_t kBitpacksPerBatch = 4096;
constexpr std::size_t kNumBitpacks = kBatchSize * kBitpacksPerBatch;

enum class pattern { random, all_ones, all_zeros };

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

template <typename bitpack_t>
std::vector<bitpack_t> make_buffer(const pattern p, const std::uint64_t seed) {
  std::vector<bitpack_t> words(kNumBitpacks);
  for (std::size_t j = 0; j < kNumBitpacks; ++j) {
    switch (p) {
      case pattern::random: words[j] = static_cast<bitpack_t>(splitmix64((seed << 32) | j)); break;
      case pattern::all_ones: words[j] = static_cast<bitpack_t>(~bitpack_t{0}); break;
      case pattern::all_zeros: words[j] = bitpack_t{0}; break;
    }
  }
  return words;
}

template <typename bitpack_t>
std::size_t host_popcount(const std::vector<bitpack_t> &words) {
  std::size_t n = 0;
  for (const auto w : words) n += static_cast<std::size_t>(std::popcount(w));
  return n;
}

}  // namespace

TEMPLATE_TEST_CASE("tally kernel counts set bits and accumulates across launches", "[mc][tally]",
                   std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t) {
  using bitpack_t = TestType;
  constexpr std::size_t kBits = sizeof(bitpack_t) * 8;

  // all_ones with a 1024-wide group is the largest per-group sum the 32-bit
  // reduction has to carry: kBits * 1024.
  const auto local = GENERATE(sycl::range<3>{1, 1, 64}, sycl::range<3>{1, 1, 1024}, sycl::range<3>{1, 4, 256});

  sycl::queue queue;
  INFO("device=" << queue.get_device().get_info<sycl::info::device::name>() << " bits=" << kBits
                 << " local={" << local[0] << "," << local[1] << "," << local[2] << "}");

  const std::array<std::vector<bitpack_t>, 3> host_buffers{
      make_buffer<bitpack_t>(pattern::random, 1),
      make_buffer<bitpack_t>(pattern::all_ones, 2),
      make_buffer<bitpack_t>(pattern::all_zeros, 3),
  };

  std::vector<bitpack_t *> device_buffers;
  for (const auto &h : host_buffers) {
    auto *d = sycl::malloc_device<bitpack_t>(kNumBitpacks, queue);
    queue.memcpy(d, h.data(), kNumBitpacks * sizeof(bitpack_t));
    device_buffers.push_back(d);
  }
  queue.wait_and_throw();

  const scram::mc::event::sample_shape<size_t_> shape{kBatchSize, kBitpacksPerBatch};
  auto block = scram::mc::event::create_tally_block<bitpack_t>(queue, device_buffers);

  using kernel_t = scram::mc::kernel::tally<prob_t, bitpack_t, size_t_>;
  const kernel_t kernel(block, shape);
  const auto nd = kernel_t::get_range(device_buffers.size(), local, shape);

  // num_one_bits accumulates across launches; total_bits is overwritten from the iteration count.
  for (const std::uint32_t iteration : {1u, 2u}) {
    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(nd, [=](sycl::nd_item<3> item) { kernel(item, iteration); });
    });
    queue.wait_and_throw();
  }

  for (std::size_t t = 0; t < host_buffers.size(); ++t) {
    INFO("tally=" << t);
    CHECK(block[t].num_one_bits == 2 * host_popcount(host_buffers[t]));
    CHECK(block[t].total_bits == 2 * kNumBitpacks * kBits);
  }

  scram::mc::event::destroy_tally_block(queue, block);
  for (auto *p : device_buffers) sycl::free(p, queue);
}
