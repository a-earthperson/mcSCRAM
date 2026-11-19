"""Unit tests for harness stats and verdicts. No GPU, no docker."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.harness.dockerutil import pin_ref, stage_file  # noqa: E402
from benchmarks.harness.inject import (  # noqa: E402
    calibrated_trials,
    choose_pass_counts,
    parse_inject_regression,
    pass_size_from_n_pair,
    pass_size_from_run,
)
from benchmarks.harness.profile import parse_nvprof, parse_nsys  # noqa: E402
from benchmarks.harness.stats import (  # noqa: E402
    bootstrap_median_ratio_ci,
    iqr,
    paired_ratios,
    z_score,
)
from benchmarks.harness.verdict import (  # noqa: E402
    VERDICT_BLOCK,
    VERDICT_ERROR,
    VERDICT_INFEASIBLE,
    VERDICT_KNOWN_FAILURE,
    VERDICT_PASS,
    VERDICT_PASS_LOW_POWER,
    VERDICT_TIMEOUT,
    ab_timeout_verdict,
    accuracy_power_verdict,
    accuracy_verdict,
    collect_timings,
    combine_verdicts,
    cpu_wall_verdict,
    finite_positive,
    gpu_kernel_verdict,
    gpu_total_verdict,
    ratio_verdict,
)


def test_z_score_zero_when_estimate_matches_ref():
    p = 1.17058e-3
    n = 1 << 20
    z = z_score(p, n, p)
    assert abs(z) < 1e-12


def test_z_score_zero_hits_uses_null_variance():
    p_ref = 9.94453321e-06
    n = 100_663_296
    z = z_score(0.0, n, p_ref)
    # |z| ≈ sqrt(n p_ref) ≈ 31.6, so a zero-hit run at 1e8 fails the 4-sigma gate.
    assert z < -4.0


def test_aa_ratio_is_pass_with_tight_noise():
    base = [10.0, 10.1, 9.9, 10.05, 9.95]
    head = [10.02, 10.08, 9.92, 10.00, 9.97]
    ratio, lo, hi = bootstrap_median_ratio_ci(base, head, n_boot=2000, seed=1)
    assert abs(ratio - 1.0) < 0.02
    assert gpu_kernel_verdict(ratio, lo, hi) == "pass"
    assert cpu_wall_verdict(ratio, lo, hi) == "pass"
    noise = iqr(paired_ratios(base, head))
    assert noise < 0.03


def test_injected_gpu_repeat_is_block():
    base = [40.0, 40.1, 39.9, 40.05, 39.95]
    head = [80.1, 80.0, 79.8, 80.2, 80.05]
    ratio, lo, hi = bootstrap_median_ratio_ci(base, head, n_boot=2000, seed=2)
    assert ratio > 1.9
    assert lo > 1.03
    assert gpu_kernel_verdict(ratio, lo, hi) == "block"


def test_injected_gpu_scale_1_10_is_block():
    base = [100.0, 100.1, 99.9, 100.05, 99.95]
    head = [110.0, 110.2, 109.8, 110.1, 109.9]
    ratio, lo, hi = bootstrap_median_ratio_ci(base, head, n_boot=3000, seed=4)
    assert 1.08 < ratio < 1.12
    assert lo > 1.03
    assert gpu_total_verdict(ratio, lo, hi) == "block"
    assert gpu_kernel_verdict(ratio, lo, hi) == "block"


def test_missing_or_zero_timing_is_error_not_crash():
    missing, reason = collect_timings(
        [{"profiler": {"total_gpu_kernel_ms": None}}], gpu=True, kind="total"
    )
    assert missing is None
    assert "missing" in reason
    zero, reason_z = collect_timings(
        [{"profiler": {"total_gpu_kernel_ms": 0.0}}], gpu=True, kind="total"
    )
    assert zero is None
    assert "non-positive" in reason_z
    cpu_zero, reason_c = collect_timings([{"sample_s": 0.0}], gpu=False)
    assert cpu_zero is None
    assert finite_positive(0.0)[0] is None
    assert combine_verdicts(["pass", VERDICT_ERROR]) == VERDICT_ERROR
    assert combine_verdicts([VERDICT_ERROR]) != VERDICT_BLOCK
    assert combine_verdicts([VERDICT_ERROR]) != VERDICT_PASS


def test_inject_scale_parses_and_multiplies_trials():
    parsed = parse_inject_regression("scale:1.10")
    assert parsed.scale == 1.10
    assert parsed.extra_runs == 0
    assert choose_pass_counts(1.10) == (10, 11)
    assert choose_pass_counts(1.05) == (20, 21)
    assert choose_pass_counts(1.10, min_passes=3) == (10, 11)
    assert choose_pass_counts(1.05, min_passes=3) == (20, 21)
    n_base, n_head = choose_pass_counts(1.10, min_passes=225)
    assert (n_base, n_head) == (230, 253)
    assert abs(n_head / n_base - 1.10) <= 0.01 * 1.10
    gpu_pass = 32 << 20
    base, head, nb, nh = calibrated_trials(gpu_pass, 1.10, min_passes=3)
    assert (nb, nh) == (10, 11)
    assert base == 10 * gpu_pass
    assert head == 11 * gpu_pass
    assert abs(head / base - 1.10) <= 0.01 * 1.10
    assert pass_size_from_run(
        {"run": {"sample_shape": {"trials_per_pass": gpu_pass}}}
    ) == gpu_pass
    assert pass_size_from_n_pair(96 << 20, 128 << 20) == gpu_pass
    try:
        pass_size_from_run({"run": {"queries": [{"n": 64}]}})
        assert False, "expected ValueError on bitpack n"
    except ValueError as exc:
        assert "bitpack" in str(exc)
    try:
        pass_size_from_run({"run": {"queries": []}})
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert parse_inject_regression("repeat").extra_runs == 1
    assert parse_inject_regression("sleep:250").sleep_ms == 250
    try:
        parse_inject_regression("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_pin_ref_is_name_at_digest():
    digest = "sha256:" + ("a" * 64)
    ref = pin_ref("ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-generic", digest)
    assert ref == f"ghcr.io/fuzzybit-org/fuzzybits-runtime@{digest}"
    already = pin_ref(ref, digest)
    assert already == ref


def test_stage_file_copies_locally(tmp_path):
    src = tmp_path / "in.xml"
    src.write_text("<opsa-mef/>\n", encoding="utf-8")
    dest_dir = tmp_path / "compat"
    out = stage_file(src, host=None, dest_dir=str(dest_dir), dest_name="model.xml")
    assert Path(out).read_text(encoding="utf-8") == "<opsa-mef/>\n"


def test_block_requires_ci_lo_above_threshold():
    # Point estimate 6% but CI lower bound 1.5% — warn, not block.
    assert gpu_kernel_verdict(1.06, 1.015, 1.11) == "warn"
    assert gpu_kernel_verdict(1.06, 1.04, 1.09) == "block"
    assert cpu_wall_verdict(1.08, 1.02, 1.14) == "warn"
    assert ratio_verdict(1.08, 1.06, 1.12, block_slowdown=0.05) == "block"
    # CPU gate is report-only until A/A CI half-width is ≤2%.
    assert cpu_wall_verdict(1.08, 1.06, 1.12) == "warn"


def test_small_significant_slowdown_is_warn():
    base = [100.0, 100.1, 99.9, 100.05, 99.95]
    head = [101.6, 101.5, 101.7, 101.55, 101.65]  # ~1.6%
    ratio, lo, hi = bootstrap_median_ratio_ci(base, head, n_boot=3000, seed=3)
    assert 1.0 < ratio < 1.03
    assert lo > 1.0
    assert gpu_kernel_verdict(ratio, lo, hi) == "warn"


def test_accuracy_failure_blocks():
    assert accuracy_verdict(4.01) == "block"
    assert accuracy_verdict(3.99) == "pass"
    assert combine_verdicts(["pass", "warn", "block"]) == "block"
    assert combine_verdicts(["pass", "inconclusive"]) == "inconclusive"
    assert combine_verdicts(["pass", "infeasible"]) == "pass"
    assert combine_verdicts(["infeasible", "infeasible"]) == "infeasible"
    assert combine_verdicts(["pass", VERDICT_KNOWN_FAILURE]) == "pass"
    assert combine_verdicts([VERDICT_KNOWN_FAILURE]) == VERDICT_KNOWN_FAILURE


def test_power_rule_infeasible_zero_hits_omits_z():
    n = 100_663_296
    judged = accuracy_power_verdict(p_ref=2.16942e-11, n=n, hits=0, z=-0.05)
    assert judged["verdict"] == VERDICT_INFEASIBLE
    assert judged["z"] is None
    assert judged["gated"] is False
    assert judged["expected_hits"] < 16


def test_power_rule_low_power_isp9607():
    n = 100_663_296
    p_ref = 9.4951e-07
    judged = accuracy_power_verdict(p_ref=p_ref, n=n, hits=91, z=-0.48)
    assert 16 <= judged["expected_hits"] < 1000
    assert judged["verdict"] == VERDICT_PASS_LOW_POWER
    assert judged["gated"] is True
    assert judged["z"] == -0.48


def test_ab_head_timeout_is_block():
    assert ab_timeout_verdict(base_timeout=False, head_timeout=True) == VERDICT_BLOCK
    assert ab_timeout_verdict(base_timeout=True, head_timeout=True) == VERDICT_TIMEOUT
    assert ab_timeout_verdict(base_timeout=True, head_timeout=False) == VERDICT_TIMEOUT
    assert ab_timeout_verdict(base_timeout=False, head_timeout=False) == VERDICT_PASS
    assert combine_verdicts(["pass", "timeout"]) == VERDICT_TIMEOUT
    assert combine_verdicts(["timeout", "block"]) == VERDICT_BLOCK


def test_power_rule_feasible_pass():
    judged = accuracy_power_verdict(
        p_ref=1.17058e-3, n=1 << 20, hits=1257, z=0.83
    )
    assert judged["verdict"] == VERDICT_PASS
    assert judged["gated"] is True
    assert judged["expected_hits"] >= 1000


def test_nvprof_parser_basic_event():
    sample = """
==1== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   89.30%  37.575ms        12  3.131ms  3.10ms  3.20ms  void basic_event(int)
                    8.00%  3.368ms         24  140.3us  130us  150us  or_gate
                    1.20%  505.0us          3  168.3us  160us  180us  tally
      API calls:   99.00%  1.000s           1  1.000s  1.000s  1.000s  cudaMalloc
"""
    parsed = parse_nvprof(sample)
    assert parsed["tool"] == "nvprof"
    assert abs(parsed["basic_event_ms"] - 37.575) < 1e-6
    assert parsed["gpu_kernel_ms"] == parsed["basic_event_ms"]
    assert parsed["primary_ms"] == parsed["total_gpu_kernel_ms"]
    assert parsed["gpu_total_ms"] == parsed["total_gpu_kernel_ms"]
    assert parsed["total_gpu_kernel_ms"] > 40.0


def test_nsys_parser_kernel_summary():
    sample = """
 CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Name
 84.100    48337000         12         4028083   4028000   4000000   4100000   10000        basic_event
 10.200    5860000          24         244166    244000    240000    250000    1000         or_gate
"""
    parsed = parse_nsys(sample)
    assert abs(parsed["basic_event_ms"] - 48.337) < 0.01
    assert parsed["total_gpu_kernel_ms"] > 50.0
