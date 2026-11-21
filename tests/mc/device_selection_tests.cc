#include <catch2/catch_test_macros.hpp>

#include "mc/queue/layer_manager.h"
#include "runtime/runtime.h"

#include <hipSYCL/runtime/settings.hpp>

#include <cstdlib>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <sycl/sycl.hpp>

namespace {

void set_mask(const char *value) {
#ifdef _WIN32
  _putenv_s("ACPP_VISIBILITY_MASK", value);
#else
  setenv("ACPP_VISIBILITY_MASK", value, 1);
#endif
}

struct restore_mask_on_exit {
  ~restore_mask_on_exit() { set_mask(""); }
};

std::set<sycl::backend> loaded_backends() {
  std::set<sycl::backend> out;
  for (const auto &d : sycl::device::get_devices()) {
    out.insert(d.get_backend());
  }
  return out;
}

// AdaptiveCpp's own name for a backend. Derived through its mask parser
// because v25.10 (the CUDA pin) has no backend_id::metal enumerator.
std::optional<std::string> mask_token(const sycl::backend backend) {
  for (const char *token : {"omp", "cuda", "hip", "ze", "ocl", "metal"}) {
    hipsycl::rt::visibility_mask_t mask;
    std::istringstream in{token};
    in >> mask;
    if (mask.count(backend) != 0) {
      return token;
    }
  }
  return std::nullopt;
}

sycl::backend mc_queue_backend() {
  return scram::mc::queue::make_throwing_sycl_queue().get_device().get_backend();
}

}  // namespace

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
  if (vis == "omp") {
    CHECK(type == sycl::info::device_type::cpu);
  } else if (vis == "cuda" || vis == "hip" || vis == "metal") {
    CHECK(type == sycl::info::device_type::gpu);
  } else {
    SUCCEED();
  }
}

// The [late-mask] cases run as mc.device_selection.late_mask, with
// ACPP_VISIBILITY_MASK unset at process start so AdaptiveCpp loads every
// backend, then set the mask the way fzb --backend does: after AdaptiveCpp
// has already read it.

TEST_CASE("MC queues select each loaded backend from a mask set after start-up",
          "[mc][device][late-mask]") {
  const restore_mask_on_exit restore;
  const auto backends = loaded_backends();
  if (backends.size() < 2) {
    SKIP("needs at least two backends visible at start-up");
  }
  for (const auto backend : backends) {
    const auto token = mask_token(backend);
    if (!token) {
      continue;
    }
    INFO("ACPP_VISIBILITY_MASK=" << *token);
    set_mask(token->c_str());
    CHECK(mc_queue_backend() == backend);
  }
}

TEST_CASE("a mask set after start-up that names no loaded backend is an error",
          "[mc][device][late-mask]") {
  const restore_mask_on_exit restore;
  set_mask("no-such-backend");
  CHECK_THROWS(scram::mc::queue::make_throwing_sycl_queue());

  const auto backends = loaded_backends();
  for (const char *token : {"hip", "ze", "cuda"}) {
    hipsycl::rt::visibility_mask_t mask;
    std::istringstream in{token};
    in >> mask;
    if (backends.count(mask.begin()->first) == 0) {
      INFO("ACPP_VISIBILITY_MASK=" << token);
      set_mask(token);
      CHECK_THROWS(scram::mc::queue::make_throwing_sycl_queue());
      return;
    }
  }
  SKIP("hip, ze and cuda are all loaded");
}

TEST_CASE("runtime::apply_backend_mask (fzb --backend) selects the MC queue's backend",
          "[mc][device][late-mask]") {
  const restore_mask_on_exit restore;
  const auto backends = loaded_backends();
  for (const char *name : {"omp", "cuda", "metal"}) {
    hipsycl::rt::visibility_mask_t mask;
    std::istringstream in{name};
    in >> mask;
    if (mask.empty() || backends.count(mask.begin()->first) == 0) {
      continue;
    }
    INFO("--backend " << name);
    fuzzybits::runtime::apply_backend_mask(name);
    CHECK(mc_queue_backend() == mask.begin()->first);
  }
  CHECK_THROWS_AS(fuzzybits::runtime::apply_backend_mask("vulkan"), fuzzybits::runtime::RuntimeError);
}
