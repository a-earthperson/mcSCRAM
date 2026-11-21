#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "mc/event/node.h"
#include "mc/event/sample_shape.h"
#include "mc/kernel/gate.h"

#include <array>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

// Widths of 128 and 256 bits are rejected at compile time; see
// bitpack_width_compile_fail.cc and the mc.compile_fail.* tests.

namespace {

using size_t_ = std::size_t;

constexpr std::size_t kNumInputs = 6;
constexpr std::size_t kBatchSize = 8;
constexpr std::size_t kBitpacksPerBatch = 4;
constexpr std::size_t kNumBitpacks = kBatchSize * kBitpacksPerBatch;

struct gate_spec {
  std::vector<std::size_t> positive;
  std::vector<std::size_t> negated;
  std::size_t at_least;
};

// Covers k = 0, k = 1, k = n, k > n, mixed polarity and all-negated inputs.
const std::vector<gate_spec> kGates{
    {{0, 1, 2}, {}, 2},
    {{0, 1, 2, 3}, {4}, 3},
    {{3}, {5}, 1},
    {{0, 1, 2, 3}, {}, 4},
    {{1, 2}, {}, 0},
    {{4}, {5}, 3},
    {{}, {0, 1, 2, 3, 4, 5}, 2},
};

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

template <typename bitpack_t>
std::array<std::vector<bitpack_t>, kNumInputs> make_inputs() {
  std::array<std::vector<bitpack_t>, kNumInputs> inputs;
  for (std::size_t i = 0; i < kNumInputs; ++i) {
    inputs[i].resize(kNumBitpacks);
    for (std::size_t j = 0; j < kNumBitpacks; ++j) {
      inputs[i][j] = static_cast<bitpack_t>(splitmix64((i << 32) | j));
    }
  }
  return inputs;
}

template <typename bitpack_t>
bitpack_t expected_word(const gate_spec &g,
                        const std::array<std::vector<bitpack_t>, kNumInputs> &inputs,
                        const std::size_t word) {
  constexpr std::size_t kBits = sizeof(bitpack_t) * 8;
  bitpack_t out = 0;
  for (std::size_t bit = 0; bit < kBits; ++bit) {
    const auto mask = static_cast<bitpack_t>(bitpack_t{1} << bit);
    std::size_t count = 0;
    for (const auto i : g.positive) count += (inputs[i][word] & mask) ? 1 : 0;
    for (const auto i : g.negated) count += (inputs[i][word] & mask) ? 0 : 1;
    if (count >= g.at_least) out = static_cast<bitpack_t>(out | mask);
  }
  return out;
}

}  // namespace

TEMPLATE_TEST_CASE("atleast gate kernel matches host k-of-n evaluation", "[mc][atleast]",
                   std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t) {
  using bitpack_t = TestType;
  const std::size_t local_y = GENERATE(1, 4, 8);

  sycl::queue queue;
  INFO("device=" << queue.get_device().get_info<sycl::info::device::name>()
                 << " bits=" << sizeof(bitpack_t) * 8 << " local_y=" << local_y);

  const auto host_inputs = make_inputs<bitpack_t>();
  std::array<bitpack_t *, kNumInputs> device_inputs{};
  for (std::size_t i = 0; i < kNumInputs; ++i) {
    device_inputs[i] = sycl::malloc_device<bitpack_t>(kNumBitpacks, queue);
    queue.memcpy(device_inputs[i], host_inputs[i].data(), kNumBitpacks * sizeof(bitpack_t));
  }
  queue.wait_and_throw();

  std::vector<std::pair<std::vector<bitpack_t *>, size_t_>> inputs_per_gate;
  std::vector<size_t_> at_least_per_gate;
  for (const auto &g : kGates) {
    std::vector<bitpack_t *> ptrs;
    for (const auto i : g.positive) ptrs.push_back(device_inputs[i]);
    for (const auto i : g.negated) ptrs.push_back(device_inputs[i]);
    inputs_per_gate.emplace_back(std::move(ptrs), g.negated.size());
    at_least_per_gate.push_back(g.at_least);
  }

  const scram::mc::event::sample_shape<size_t_> shape{kBatchSize, kBitpacksPerBatch};
  auto block = scram::mc::event::create_atleast_gate_block<bitpack_t, size_t_>(
      queue, inputs_per_gate, at_least_per_gate, shape.num_bitpacks());

  using kernel_t = scram::mc::kernel::op<scram::core::Connective::kAtleast, bitpack_t, size_t_>;
  const kernel_t kernel(block, shape);
  const auto nd = kernel_t::get_range(kGates.size(), sycl::range<3>{1, local_y, 1}, shape);
  queue.submit([&](sycl::handler &cgh) { cgh.parallel_for(nd, kernel); });
  queue.wait_and_throw();

  std::vector<bitpack_t> out(kGates.size() * kNumBitpacks);
  queue.memcpy(out.data(), block.buffers, out.size() * sizeof(bitpack_t)).wait_and_throw();

  std::size_t mismatches = 0;
  for (std::size_t g = 0; g < kGates.size(); ++g) {
    for (std::size_t w = 0; w < kNumBitpacks; ++w) {
      const auto want = expected_word(kGates[g], host_inputs, w);
      const auto got = out[g * kNumBitpacks + w];
      if (got != want && ++mismatches <= 8) {
        UNSCOPED_INFO("gate=" << g << " word=" << w << std::hex
                              << " got=0x" << static_cast<std::uint64_t>(got)
                              << " want=0x" << static_cast<std::uint64_t>(want));
      }
    }
  }
  CHECK(mismatches == 0);

  scram::mc::event::destroy_atleast_gate_block(queue, block);
  for (auto *p : device_inputs) sycl::free(p, queue);
}
