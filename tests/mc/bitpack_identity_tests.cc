#include <catch2/catch_test_macros.hpp>

#include "mc/event/host_threshold.h"
#include "mc/event/node.h"
#include "mc/kernel/basic_event.h"
#include "mc/prng/bernoulli.h"
#include "mc/prng/philox128.h"
#include "mc/prng/state128.h"
#include "pcfb/threshold.h"

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
using scram::mc::prng::bernoulli_threshold;
using scram::mc::prng::kLowWordGenerationBase;
using scram::mc::prng::less_than_threshold;
using scram::mc::prng::threshold_from_probability;

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
//
// #17 two-word compare is identical to the previous 32-bit `u_hi < T_hi`
// sampler except on high-word ties (u_hi == T_hi), which have probability
// 2^-32 per bit. These 32 packs contain 2048 bits, so a checksum change
// here is a real stream/tie event and must be justified in the PR, not
// silently re-baselined.
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

scram::mc::prng::state128 seed_for(std::uint32_t event_idx, std::uint32_t pdag_idx,
                                  std::uint32_t sample_idx) {
  return scram::mc::prng::state128{
      .x = {
          pdag_idx + 1,
          event_idx + 1,
          sample_idx + 1,
          (kIteration + 1) << 6,
      },
  };
}

std::uint64_t host_checksum() {
  std::vector<bitpack_t> packs;
  packs.reserve(kNumEvents * kBatchSize * kBitpacksPerBatch);
  for (const auto &[event_idx, pdag_idx, p] : kEvents) {
    const auto threshold = threshold_from_probability(p);
    for (std::uint32_t batch_idx = 0; batch_idx < kBatchSize; ++batch_idx) {
      for (std::uint32_t bitpack_idx = 0; bitpack_idx < kBitpacksPerBatch; ++bitpack_idx) {
        const std::uint32_t sample_idx = batch_idx * kBitpacksPerBatch + bitpack_idx;
        packs.push_back(scram::mc::prng::philox::pack_bernoulli_draws<bitpack_t>(
            seed_for(event_idx, pdag_idx, sample_idx), threshold));
      }
    }
  }
  return fnv1a64(packs.data(), packs.size());
}

using block_t = scram::mc::event::basic_event_block<prob_t, bitpack_t>;
using kernel_t = scram::mc::kernel::basic_event<prob_t, bitpack_t, size_t_>;
using fixup_t = scram::mc::kernel::basic_event_tie_fixup<prob_t, bitpack_t, size_t_>;
using shape_t = scram::mc::event::sample_shape<size_t_>;

// Sampler launch followed by the tie fix-up, as the layer manager submits them.
void run_sampler(sycl::queue &queue, block_t &block, const shape_t &shape, const std::size_t num_events,
                 const sycl::range<3> &local) {
  const auto nd = kernel_t::get_range(num_events, local, shape);
  const auto slots = kernel_t::allocate_tie_slots(queue, nd);
  block.tie_slots = slots.data;
  const kernel_t kernel(block, shape, slots);
  const fixup_t fixup(block, shape, slots, nd);
  const auto fixup_nd =
      fixup_t::get_range(slots, queue.get_device().get_info<sycl::info::device::max_work_group_size>());
  const auto sampled = queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(nd, [=](sycl::nd_item<3> item) { kernel(item, kIteration); });
  });
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(sampled);
    cgh.parallel_for(fixup_nd, [=](sycl::nd_item<1> item) { fixup(item, kIteration); });
  });
  queue.wait_and_throw();
}

std::uint64_t device_checksum(sycl::queue &queue) {
  std::vector<std::pair<std::int32_t, bernoulli_threshold>> indexed;
  indexed.reserve(kNumEvents);
  for (const auto &[event_idx, pdag_idx, p] : kEvents) {
    (void)event_idx;
    indexed.emplace_back(static_cast<std::int32_t>(pdag_idx), threshold_from_probability(p));
  }

  const shape_t shape{kBatchSize, kBitpacksPerBatch};
  block_t block = scram::mc::event::create_basic_event_block<prob_t, bitpack_t>(
      queue, indexed, shape.num_bitpacks());
  run_sampler(queue, block, shape, kNumEvents, sycl::range<3>{1, 1, 1});

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

TEST_CASE("two-word compare matches 32-bit sampler except on high-word ties", "[mc][bit-identity][ties]") {
  const auto t = threshold_from_probability(0.1);
  REQUIRE(t.always_true == 0);
  REQUIRE(t.hi == static_cast<std::uint32_t>(0.1 * 4294967296.0));

  CHECK(less_than_threshold(t.hi - 1, 0, t));
  CHECK_FALSE(less_than_threshold(t.hi + 1, 0, t));
  CHECK_FALSE(less_than_threshold(t.hi, t.lo, t));
  if (t.lo > 0) {
    CHECK(less_than_threshold(t.hi, t.lo - 1, t));
  }

  const auto always = threshold_from_probability(1.0);
  REQUIRE(always.always_true == 1);
  CHECK(less_than_threshold(0xffffffffu, 0xffffffffu, always));

  const auto zero = threshold_from_probability(0.0);
  REQUIRE(zero.hi == 0);
  REQUIRE(zero.lo == 0);
  CHECK_FALSE(less_than_threshold(0, 0, zero));
}

TEST_CASE("p=1e-12 has a non-zero 64-bit threshold and fires on a forced high-word tie",
          "[mc][rare]") {
  constexpr double kP = 1e-12;
  REQUIRE(kP < 0x1p-32);
  REQUIRE(static_cast<std::uint64_t>(kP * 4294967296.0) == 0);

  const auto t = threshold_from_probability(kP);
  const auto T = pcfb::probability_to_threshold(kP);
  REQUIRE(t.always_true == 0);
  REQUIRE(t.hi == 0);
  REQUIRE(t.lo != 0);
  REQUIRE(t.lo == static_cast<std::uint32_t>(T));
  REQUIRE((T >> 32U) == 0);

  // Predicate: hits only on u_hi == 0 and u_lo < T_lo.
  CHECK_FALSE(less_than_threshold(0, t.lo, t));
  CHECK(less_than_threshold(0, t.lo - 1, t));
  CHECK_FALSE(less_than_threshold(1, 0, t));
  CHECK(less_than_threshold(0, 0, t));

  // Forced-counter: take a real Philox block, set T_hi to lane 0's high word
  // so the sampler must draw u_lo from generation+32, and check the bit.
  const auto seed = seed_for(0, 2, 0);
  scram::mc::prng::state128 hi{};
  scram::mc::prng::philox::generate(&seed, &hi, 0);
  bernoulli_threshold forced = t;
  forced.hi = hi.x[0];
  scram::mc::prng::state128 lo{};
  scram::mc::prng::philox::generate(&seed, &lo, kLowWordGenerationBase);
  const auto pack = scram::mc::prng::philox::sample<bitpack_t>(&seed, forced, 0);
  const bool expected = lo.x[0] < forced.lo;
  CHECK(((pack >> 0) & 1ull) == static_cast<bitpack_t>(expected));
}

TEST_CASE("device stores two-word threshold for p=1e-12", "[mc][rare][device]") {
  const auto t = threshold_from_probability(1e-12);
  std::vector<std::pair<std::int32_t, bernoulli_threshold>> indexed{{2, t}};
  const shape_t shape{1, 1};

  sycl::queue queue;
  block_t block = scram::mc::event::create_basic_event_block<prob_t, bitpack_t>(
      queue, indexed, shape.num_bitpacks());
  REQUIRE(block.data[0].probability_threshold.hi == 0);
  REQUIRE(block.data[0].probability_threshold.lo == t.lo);
  REQUIRE(block.data[0].probability_threshold.always_true == 0);

  run_sampler(queue, block, shape, 1, sycl::range<3>{1, 1, 1});

  bitpack_t host = 0;
  queue.memcpy(&host, block.buffers, sizeof(bitpack_t)).wait_and_throw();
  const auto expected = scram::mc::prng::philox::pack_bernoulli_draws<bitpack_t>(
      seed_for(0, 2, 0), t);
  CHECK(host == expected);
  scram::mc::event::destroy_basic_event_block(queue, block);
}

TEST_CASE("device tie fix-up matches the exact two-word sampler on forced ties", "[mc][rare][device][ties]") {
  using scram::mc::prng::state128;
  namespace philox = scram::mc::prng::philox;

  // (event_idx, sample_idx, generation, lane, u_lo < T_lo). Event i has
  // pdag_idx i + 2. With local {2, 1, 16} two events share each 32-lane
  // sub-group, so events 2 and 3 tie inside one sub-group record.
  struct forced_tie {
    std::uint32_t event_idx;
    std::uint32_t sample_idx;
    std::uint32_t generation;
    std::uint32_t lane;
    bool hit;
  };
  constexpr std::array<forced_tie, 4> kForced{{
      {0, 5, 0, 0, true},
      {1, 37, 3, 2, false},
      {2, 3, 7, 1, true},
      {3, 12, 15, 3, true},
  }};
  const shape_t shape{2, 32};
  const sycl::range<3> local{2, 1, 16};

  std::vector<std::pair<std::int32_t, bernoulli_threshold>> indexed;
  for (const auto &f : kForced) {
    const auto seed = seed_for(f.event_idx, f.event_idx + 2, f.sample_idx);
    state128 hi{};
    state128 lo{};
    philox::generate(&seed, &hi, static_cast<std::uint8_t>(f.generation));
    philox::generate(&seed, &lo, static_cast<std::uint8_t>(f.generation + kLowWordGenerationBase));
    REQUIRE(lo.x[f.lane] != 0xffffffffu);
    bernoulli_threshold t{};
    t.hi = hi.x[f.lane];
    t.lo = f.hit ? lo.x[f.lane] + 1u : lo.x[f.lane];
    indexed.emplace_back(static_cast<std::int32_t>(f.event_idx + 2), t);
  }
  // always_true never records a tie, and event 5 is out of bounds (x rounds up to 6).
  indexed.emplace_back(6, threshold_from_probability(1.0));
  const std::size_t num_events = indexed.size();

  for (const auto &f : kForced) {
    const auto &t = indexed[f.event_idx].second;
    const auto seed = seed_for(f.event_idx, f.event_idx + 2, f.sample_idx);
    bool any_tie = false;
    const auto hi_only = philox::pack_bernoulli_draws_hi<bitpack_t>(seed, t, any_tie);
    const auto exact = philox::pack_bernoulli_draws<bitpack_t>(seed, t);
    const bitpack_t bit = bitpack_t(1) << (f.generation * 4 + f.lane);
    REQUIRE(any_tie);
    CHECK((hi_only & bit) == 0);
    CHECK(((exact & bit) != 0) == f.hit);
  }

  sycl::queue queue;
  INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());
  block_t block = scram::mc::event::create_basic_event_block<prob_t, bitpack_t>(
      queue, indexed, shape.num_bitpacks());
  run_sampler(queue, block, shape, num_events, local);

  const std::size_t n = num_events * shape.num_bitpacks();
  std::vector<bitpack_t> device(n);
  queue.memcpy(device.data(), block.buffers, n * sizeof(bitpack_t)).wait_and_throw();
  scram::mc::event::destroy_basic_event_block(queue, block);

  std::size_t mismatches = 0;
  for (std::uint32_t e = 0; e < num_events; ++e) {
    for (std::uint32_t s = 0; s < shape.num_bitpacks(); ++s) {
      const auto expected =
          philox::pack_bernoulli_draws<bitpack_t>(seed_for(e, e + 2, s), indexed[e].second);
      if (device[e * shape.num_bitpacks() + s] != expected) {
        ++mismatches;
        UNSCOPED_INFO("event=" << e << " sample=" << s << " device=0x" << std::hex
                               << device[e * shape.num_bitpacks() + s] << " expected=0x" << expected
                               << std::dec);
      }
    }
  }
  CHECK(mismatches == 0);
}
