#pragma once
#include <algorithm>
#include <cmath>

namespace scram::mc::event::detail {

struct lr_result {
    std::double_t p_clamped;   // probability after clamping (for threshold)
    std::double_t lr1;         // likelihood ratio for bit = 1
    std::double_t lr0;         // likelihood ratio for bit = 0
};

// Clamp user-supplied probability p and bias q to numerically safe ranges and
// return the corresponding importance-sampling likelihood ratios.
[[nodiscard]] inline lr_result
safe_likelihood_ratios(std::double_t p, std::double_t q) {
    constexpr std::double_t eps = 1e-12;

    p = std::clamp(p, 0.0, 1.0);
    q = std::clamp(q, eps, 1.0 - eps);

    if (p >= 1.0 - eps) p = 1.0 - eps;
    if (p <= eps)       p = eps;

    lr_result out;
    out.p_clamped = p;
    out.lr1       = p / q;
    out.lr0       = (1.0 - p) / (1.0 - q);
    return out;
}

} // namespace scram::mc::event::detail
