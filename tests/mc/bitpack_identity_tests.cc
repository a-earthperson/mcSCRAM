#include <catch2/catch_test_macros.hpp>

#include "mc/event/node.h"
#include "mc/kernel/basic_event.h"
#include "mc/prng/philox128.h"
#include "mc/prng/state128.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <ios>
#include <sycl/sycl.hpp>
#include <tuple>
#include <vector>

namespace {

using bitpack_t = std::uint64_t;
using prob_t = double;
using size_t_ = std::size_t;

constexpr std::size_t kNumEvents = 4;
constexpr std::size_t kBatchSize = 4;
constexpr std::size_t kBitpacksPerBatch = 2;
constexpr std::uint32_t kIteration = 0;

// (event_idx, pdag_idx, p)
constexpr std::array<std::tuple<std::uint32_t, std::uint32_t, double>, kNumEvents> kEvents{{
    {0, 2, 0.1},
    {1, 3, 0.5},
    {2, 4, 0.0},
    {3, 5, 1.0},
}};

// FNV-1a 64 of the 32 little-endian uint64 bitpacks produced by host Philox
// at the counters above. CPU (omp) and GPU (cuda) must match this value for
// a counter-based Philox; a sampler refactor (#16) that changes the stream
// will fail this test on every backend.
constexpr std::uint64_t kExpectedChecksum = 0x113d94c03a83d4aaull;

std::uint64_t fnv1a64(const bitpack_t *data, std::size_t n) {
  std::uint64_t h = 0xcbf29ce484222325ull;
  for (std::size_t i = 0; i < n; ++i) {
    unsigned char bytes[8];
    std::memcpy(bytes, &data[i], 8);
    for (unsigned char b : bytes) {
      h ^= b;
      h *= 0x100000001b3ull;
    }
  }
  return h;
}

std::uint64_t host_checksum() {
  constexpr double two_to_32 = static_cast<double>(UINT64_C(1) << 32);
  std::vector<bitpack_t> packs;
  packs.reserve(kNumEvents * kBatchSize * kBitpacksPerBatch);
  for (const auto &[event_idx, pdag_idx, p] : kEvents) {
    const auto threshold = static_cast<std::uint64_t>(p * two_to_32);
    for (std::uint32_t batch_idx = 0; batch_idx < kBatchSize; ++batch_idx) {
      for (std::uint32_t bitpack_idx = 0; bitpack_idx < kBitpacksPerBatch; ++bitpack_idx) {
        const std::uint32_t sample_idx = batch_idx * kBitpacksPerBatch + bitpack_idx;
        const scram::mc::prng::state128 seed_base{
            .x = {
                pdag_idx + 1,
                event_idx + 1,
                sample_idx + 1,
                (kIteration + 1) << 6,
            },
        };
        packs.push_back(
            scram::mc::prng::philox::pack_bernoulli_draws<bitpack_t>(seed_base, threshold));
      }
    }
  }
  return fnv1a64(packs.data(), packs.size());
}

std::uint64_t device_checksum(sycl::queue &queue) {
  using block_t = scram::mc::event::basic_event_block<prob_t, bitpack_t>;
  using kernel_t = scram::mc::kernel::basic_event<prob_t, bitpack_t, size_t_>;

  std::vector<std::pair<std::int32_t, prob_t>> indexed;
  indexed.reserve(kNumEvents);
  for (const auto &[event_idx, pdag_idx, p] : kEvents) {
    (void)event_idx;
    indexed.emplace_back(static_cast<std::int32_t>(pdag_idx), p);
  }

  const scram::mc::event::sample_shape<size_t_> shape{kBatchSize, kBitpacksPerBatch};
  block_t block = scram::mc::event::create_basic_event_block<prob_t, bitpack_t>(
      queue, indexed, shape.num_bitpacks());

  kernel_t kernel(block, shape);
  const sycl::range<3> local{1, 1, 1};
  const auto nd = kernel.get_range(kNumEvents, local, shape);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(nd, [=](sycl::nd_item<3> item) { kernel(item, kIteration); });
  });
  queue.wait_and_throw();

  const std::size_t n = kNumEvents * shape.num_bitpacks();
  std::vector<bitpack_t> host(n);
  queue.memcpy(host.data(), block.buffers, n * sizeof(bitpack_t)).wait_and_throw();

  const auto checksum = fnv1a64(host.data(), n);
  scram::mc::event::destroy_basic_event_block(queue, block);
  return checksum;
}

}  // namespace

TEST_CASE("basic_event bitpacks are bit-identical at fixed Philox counters", "[mc][bit-identity]") {
  const auto host = host_checksum();
  INFO("host checksum=0x" << std::hex << host);
  REQUIRE(host == kExpectedChecksum);

  sycl::queue queue;
  INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());
  const auto device = device_checksum(queue);
  INFO("device checksum=0x" << std::hex << device);
  CHECK(device == kExpectedChecksum);
  CHECK(device == host);
}
