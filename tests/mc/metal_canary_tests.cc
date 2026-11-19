#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <sycl/sycl.hpp>

// Removal canaries for src/mc/compat/metal.h. On Metal these exercise the
// native constructs the shims paper over. macos-metal registers them
// WILL_FAIL (xfail-strict): when AdaptiveCpp lowers the construct, CI flips
// and the shim is deleted. Do not label these `mc`; default presets exclude
// `metal-canary`. Compiles on v25.10.0 (no sycl::backend::metal enumerator).

TEST_CASE("native 64-bit reduce_over_group on Metal", "[metal-canary][reduce64]") {
    sycl::queue queue;
    auto *out = sycl::malloc_shared<std::size_t>(1, queue);
    REQUIRE(out != nullptr);
    *out = 0;
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> item) {
            const std::size_t local = 1;
            const std::size_t sum = sycl::reduce_over_group(item.get_group(), local, sycl::plus<>());
            if (item.get_local_linear_id() == 0) {
                *out = sum;
            }
        });
    });
    queue.wait_and_throw();
    REQUIRE(*out == 32);
    sycl::free(out, queue);
}

TEST_CASE("native double in a Metal kernel", "[metal-canary][double]") {
    sycl::queue queue;
    auto *out = sycl::malloc_shared<double>(1, queue);
    REQUIRE(out != nullptr);
    *out = 0.0;
    queue.submit([&](sycl::handler &cgh) {
        cgh.single_task([=] {
            const double half = 0.5;
            *out = half * half;
        });
    });
    queue.wait_and_throw();
    REQUIRE(*out == 0.25);
    sycl::free(out, queue);
}
