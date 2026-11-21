#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>
#include <sycl/sycl.hpp>

TEST_CASE("default SYCL queue honours ACPP_VISIBILITY_MASK", "[mc][device]") {
  sycl::queue queue;
  const auto type = queue.get_device().get_info<sycl::info::device::device_type>();
  const auto name = queue.get_device().get_info<sycl::info::device::name>();
  const char *mask = std::getenv("ACPP_VISIBILITY_MASK");
  INFO("ACPP_VISIBILITY_MASK=" << (mask ? mask : "(unset)") << " device=" << name);

  if (mask == nullptr || *mask == '\0') {
    SUCCEED();
    return;
  }
  const std::string vis{mask};
  const int backend = static_cast<int>(queue.get_device().get_backend());
  INFO("sycl::backend=" << backend);
  if (vis == "omp") {
    CHECK(type == sycl::info::device_type::cpu);
  } else if (vis == "cuda" || vis == "hip") {
    CHECK(type == sycl::info::device_type::gpu);
  } else if (vis == "metal") {
    CHECK(type == sycl::info::device_type::gpu);
    CHECK(backend == 5);
  } else {
    SUCCEED();
  }
}
