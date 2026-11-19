# Benchmarks

Frozen accuracy and performance measurements for the Monte Carlo engine
([#4](https://github.com/fuzzybit-org/fuzzybits/issues/4)). This is the
dataset ([#120](https://github.com/fuzzybit-org/fuzzybits/issues/120)) and
harness ([#121](https://github.com/fuzzybit-org/fuzzybits/issues/121)). The PR
gate that calls the harness is [#122](https://github.com/fuzzybit-org/fuzzybits/issues/122);
published history is [#123](https://github.com/fuzzybit-org/fuzzybits/issues/123).

Legacy scripts under `benchmarks/` (`bench_scram.py`, `benchmark_compiler.py`,
…) are from the mcSCRAM era. They stay in the tree; the harness does not use
them.

## Dataset versioning

`benchmarks/datasets/v1/` is **frozen**. A dataset version never changes once
used: the harness, the PR gate and the published history pin a dataset id.
Edits to membership, references, trial counts, tags or the schema are a new
version (`v2`, …).

Regenerate only when cutting a new version:

```bash
python -m benchmarks.datasets.freeze
```

## Dataset v1

| set | files | role |
|---|---|---|
| Aralia (`input/Aralia`, 43 XML) | accuracy (all 43) and a `perf` subset | exact top-event probabilities |
| generic-pwr-model (`input/generic-pwr-model`, 95 OpenPSA event-tree-group XML, CC0, submodule `ce2ef006…`) | `pwr`: compile / PDAG size / throughput | SAPHIRE MCS references are **not** used for accuracy |
| `synthetic-models` | out of scope | no license declared |

Each model records: path, sha256 of the XML, submodule commit (PWR only), the
query (top event / gate id), the reference probability and its source, tags
(`accuracy`, `perf`, `hard`, `pwr`), trial counts, and a time budget.

### Accuracy references

42 Aralia models use the exact BDD values in
`tools/server/src/fuzzybits_server/model_oracle_p.json` (`source: bdd`).

`nus9601` is not in that file. The v1 reference is **P = 9.94453321e-06**,
computed with nombrillo by Kirill Shumilov (`@kshumilov`), recorded as an
**exact** reference. `fzb inspect --exact` refuses 1,567 SAMPLE events (cap
30). `scram-cli --bdd` / `--zbdd` runs out of memory on this model; do not
retry that path.

The compiled query for every Aralia model is gate `r1` (the first
`define-gate`, which is the fault-tree top).

### Trial counts and power rule

Accuracy runs target **n·p ≥ 1000** so a 4-sigma check is meaningful, then
round up to a multiple of 2^20 (the engine pass size) and cap at
`fzb --trials 1e8` = 96·2^20 = 100,663,296.

The harness judges **expected hits = n·p_ref first**, then z. Thresholds live
in `manifest.defaults` (`accuracy_infeasible_hits`, `accuracy_low_power_hits`):

| expected hits n·p_ref | verdict | counts as an accuracy pass? |
|---|---|---|
| < `accuracy_infeasible_hits` (**16**) or p_ref < 2^-32 (~2.33e-10, [#17](https://github.com/fuzzybit-org/fuzzybits/issues/17)) | **`infeasible`** | no |
| 16 ≤ n·p_ref < `accuracy_low_power_hits` (**1000**) and \|z\| ≤ 4 | **`pass (low power)`** | yes, annotated |
| n·p_ref ≥ 1000 and \|z\| ≤ 4 | **`pass`** | yes |
| n·p_ref ≥ 16 and \|z\| > 4 | **`block`** | no |

16 is the vacuous-pass bound: 0 hits gives \|z\| = √(n p_ref) < 4, so the test
has no power. (The “e.g. 100” figure is the same order of magnitude;
`isp9607` at n·p ≈ 96 is therefore `pass (low power)`, not infeasible.) When
hits = 0 the table omits z; the power rule has already classified the row.

Rare-event work is [#6](https://github.com/fuzzybit-org/fuzzybits/issues/6)
(importance sampling) and [#17](https://github.com/fuzzybit-org/fuzzybits/issues/17)/[#91](https://github.com/fuzzybit-org/fuzzybits/issues/91).
GPUs help by a few orders of magnitude and still do not reach n·p ≈ 1000 for
these models. Do not enlarge the CI budget to chase them.

| model | p | accuracy n | expected hits | verdict category |
|---|---|---|---|---|
| most Aralia | ≳ 1e-4 | 2^20 | ≫ 1000 | `pass` |
| isp9605 | 1.37e-5 | 70·2^20 | ~1000 | `pass` |
| **nus9601** | 9.94453321e-06 | **96·2^20 (1e8)** | **~1001** | `pass` (CPU budget **180 s**) |
| isp9607 | 9.50e-7 | 1e8 cap | ~96 | `pass (low power)` |
| das9205 | 1.38e-8 | 1e8 cap | ~1.4 | `infeasible` (n·p ≪ 16) |
| das9204, das9209, edf9206 | ≲ 2e-11 | 1e8 cap | ≪ 1 | `infeasible` (n·p ≪ 16 **and** p < 2^-32, #17) |

z, when shown: |z| ≤ 4 with σ̂² = p̂(1−p̂)/n + σ_ref² (σ_ref = 0 for exact
references). If p̂ ∈ {0,1} on a **gated** row, the null variance
p_ref(1−p_ref)/n is used so the arithmetic is defined; those rows should
already have been classified infeasible.

### Performance subset

Chosen for coverage of the engine's cost profile (kernel breakdowns from
[PR #91](https://github.com/fuzzybit-org/fuzzybits/pull/91) and
[#20](https://github.com/fuzzybit-org/fuzzybits/issues/20)):

| model | reason |
|---|---|
| `das9201` | Sampler-dominated: on V100S at 1e8, `basic_event` is ~89% of GPU kernel time (37.6 of 42.1 ms). |
| `das9701` | Largest Aralia gate counts; fused-kernel prototype's best case in #20 (3.11×). |
| `nus9601` | 1,567 basic events; fused BE tile does not fit in shared memory (#20). |
| `edf9203` | Known fused-path regressor in #20 (2.2× slower than unfused). |
| `chinese` | Small model; launch / JIT / host overhead baseline (~9–13 ms kernels). |

Perf trial count is 1e8 on GPU. CPU perf trial counts are sized per model so
in-process sampling is ~2–5 s (`trials.cpu_perf` in the manifest). GPU metrics
are per-kernel times (`basic_event`, gate kernels, tally) via **nvprof** on
V100S (`sm_70`) and **nsys** on RTX 3070 (`sm_86`). Warm and cold AdaptiveCpp
cache times are reported separately; the gate uses **warm** numbers.

The A/B **gate metric is `gpu_total_ms`** (all compute kernels; memcpy
excluded), not `basic_event` alone. On das9701 at 1e8, `basic_event` is 81.5 ms
of 218 ms total GPU kernel time (~37%). A kernel-only gate would miss
regressions in gates/tally/other, and a 10% `basic_event` regression is only
~3.7% of total — right on the 3% threshold. The harness therefore **dual-gates**:
it blocks if *either* `gpu_total_ms` or `gpu_kernel_ms` (`basic_event`) has
`ci_lo > 1.03`. `gpu_kernel_ms` is skipped when `basic_event` is missing, not
treated as zero.

### generic-pwr-model

v1 records compile time, PDAG size (`fzb inspect --json`: tensors, operators,
samples) and sampling throughput. Accuracy is deferred: SAPHIRE MCS values are
mostly 1e-10 to 1e-6, values below 2.3e-10 quantize to zero ([#17](https://github.com/fuzzybit-org/fuzzybits/issues/17)),
and only the first fault-tree top event is compiled
([#92](https://github.com/fuzzybit-org/fuzzybits/issues/92)). Revisit with #92,
#17 and #6. The generic-pwr-model OpenPSA export is not self-contained MEF;
the harness rewrites a cached copy via `benchmarks/datasets/compat/generic_pwr.py`.

A fresh clone needs:

```bash
git submodule update --init input/generic-pwr-model
```

## Loader

```bash
python -m benchmarks validate
```

Checks: JSON structure, tags, references in (0, 1), files exist, sha256
matches, PWR submodule pin. This is the CI test (`tests/python/test_benchmark_dataset.py`);
it is CPU-only and does not import the native engine.

## Harness

```text
python -m benchmarks validate
python -m benchmarks run  --target cpu|v100s|rtx3070 --mode accuracy|perf|compile \
    --image IMG --out results.json [--md summary.md] [--ids chinese das9201]
python -m benchmarks ab   --target cpu|v100s|rtx3070 --rounds 6 \
    --base SHA --head SHA --build \
    --out ab.json [--md ab.md]
python -m benchmarks ab   --target v100s --base-image IMG --head-image IMG \
    --ids chinese --inject-regression scale:1.10 --out aa.json
python -m benchmarks summarize results.json
```

`run` and `ab` execute `fzb compile` / `inspect` / `run` **inside** a runtime
image. Models are mounted from the docker host (`/tmp/fuzzybits-bench/input` by
default). Results are versioned JSON (`benchmarks/schemas/results.v1.json`)
plus a Markdown table suitable for a PR comment.

### A/B mode (what the PR gate uses)

Numbers only compare at the few-percent level when **both commits are built
with the same recipe and timed alternately on the same host** (lesson from
[PR #91](https://github.com/fuzzybit-org/fuzzybits/pull/91#issuecomment-7019747119)).

1. Build base and head with `docker/runtime.Dockerfile --target runtime` on
   `ghcr.io/fuzzybit-org/fuzzybits-toolchain:v25.10.0-generic-<sha256(toolchain.env)[:12]>`.
   Context is `git archive <sha> | docker build -`. On macOS set
   `COPYFILE_DISABLE=1` (AppleDouble `._*` files broke corpus tests).
   Registry tags are resolved **once** with `docker buildx imagetools inspect`
   and then run as `name@sha256:<64 hex>`. Do not pass a mutable tag such as
   `nightly-generic` through to `docker run`: gateway and xerxes previously
   resolved that tag to different local config ids
   (`sha256:bb0239034e17…` vs `sha256:26b0c066974e…`). Results JSON records the
   full digest.
2. Warm both AdaptiveCpp JIT caches (`ACPP_APPDB_DIR` on a per-side volume).
3. Alternate base/head for N rounds (default 6; CPU A/A uses 10). Drop the first round.
4. CPU: `OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores`,
   `docker --cpuset-cpus 0-7`, `ACPP_VISIBILITY_MASK=omp`. The metric is
   **in-process `fuzzybits.run` time after AdaptiveCpp init and a 2^20 JIT
   warmup** (`cpu_sample_s`). Do not use container wall time: on chinese at
   1e8, container wall was ~1.2 s while sampling is ~1.3 s once init is
   excluded — still too short, so CPU perf trial counts are sized for
   **~2–5 s of sampling**. GPU: `ACPP_VISIBILITY_MASK=cuda`. V100S uses
   `nvprof --print-gpu-summary --profile-child-processes` (volume `pr91-nvprof`
   at `/opt/cuda`, binary `/opt/cuda/cuda/bin/nvprof`); RTX 3070 uses
   `nsys profile --stats=true --trace-fork-before-exec=true` (volume `pr91-data`).
5. Report median ratio head/base with a bootstrap 95% CI.
6. Verdict:
   - **block** only when the CI *lower bound* exceeds the threshold: GPU
     `gpu_total_ms` **or** `gpu_kernel_ms` `ci_lo > 1.03`. CPU sample
     `ci_lo > 1.05` is computed the same way but is **report-only**
     (`CPU_GATE_BLOCKS = False`, [#124](https://github.com/fuzzybit-org/fuzzybits/issues/124)).
     A point estimate above the threshold with `ci_lo` still below it is
     **warn**, not block. Accuracy |z| > 4 always blocks.
   - **warn** for a significant slowdown whose CI still overlaps the
     threshold. CPU above-threshold is always warn, never block.
   - **error** if a gated timing is missing or ≤ 0 (never a crash, never
     `block` / `pass`).
   - **pass** otherwise.
   - **inconclusive** if idle/load guards fail, rather than reporting noise as
     a regression.

### A/A noise (nightly-generic, identical images)

`nightly-generic` must be pinned by digest before A/A. Local config ids were
host-specific; the registry manifest is the pin.

| target | model | metric | rounds kept | ratio | 95% CI | half-width | IQR | verdict |
|---|---|---|---|---|---|---|---|---|
| V100S | chinese | gpu_kernel_ms (then) | 5 | 1.004 | [0.991, 1.013] | 1.1% | 0.47% | pass |
| CPU (container wall, chinese 1e8) | chinese | cpu_wall_s | 5 | 1.089 | [0.887, 1.148] | 13% | 17% | pass (unusable) |
| CPU | chinese | cpu_sample_s | 9 | 0.999 | [0.963, 1.027] | 3.2% | 4.6% | pass (report-only) |
| CPU | das9201 | cpu_sample_s | 9 | 1.011 | [0.977, 1.035] | 2.9% | 2.8% | pass (report-only) |
| CPU | das9701 | cpu_sample_s | 9 | 1.040 | [0.964, 1.097] | 6.6% | 8.9% | pass (report-only) |

Sampling times were in band (chinese ~2.9 s at 225·2^20, das9201 ~3.2 s at
80·2^20, das9701 ~3.5 s at 16·2^20) with `OMP_NUM_THREADS=8`,
`--cpuset-cpus 0-7`. CPU half-widths are **2.9–6.6%**, not ≤2%. Gateway load
rose to 10 during the run (ZFS + the 8-thread jobs). The CPU gate is
**report-only** (`CPU_GATE_BLOCKS = False`, [#124](https://github.com/fuzzybit-org/fuzzybits/issues/124)): it can warn, it does not block.

Pinned-image A/A and `scale:1.10` (registry digest
`sha256:97e5ffe141ce4f1a969f9a520264f940b0e5c256b9cf7780dfd6e2a9831d7803`,
`ghcr.io/fuzzybit-org/fuzzybits-runtime@sha256:97e5ffe1…`):

| target | model | inject | metric | ratio | 95% CI | half-width | verdict |
|---|---|---|---|---|---|---|---|
| V100S | chinese | A/A | gpu_total_ms | 1.002 | [0.990, 1.007] | 0.87% | **pass** |
| V100S | chinese | A/A | gpu_kernel_ms | 1.001 | [0.989, 1.008] | 0.97% | pass (dual-gate) |
| V100S | chinese | scale:1.10 | gpu_total_ms | 1.317 | [1.302, 1.326] | | **block** |
| V100S | chinese | scale:1.10 | gpu_kernel_ms | 1.316 | [1.298, 1.325] | | block (dual-gate) |
| CPU | chinese | A/A | cpu_sample_s | 0.993 | [0.977, 1.029] | 2.6% | pass (report-only) |
| CPU | chinese | scale:1.10 | cpu_sample_s | 1.020 | [0.861, 1.171] | 15.5% | pass (report-only; CI includes 1) |

`scale:1.10` multiplies the requested trial count and rounds up to 2^20.
On V100S chinese the GPU pass is 32·2^20, so 1.10× of 96·2^20 (3 passes)
becomes 4 passes (128·2^20, 1.33×). `gpu_total_ms` ratio 1.317 still
clears `ci_lo > 1.03`. That is the small-regression check; `repeat` remains
the 2× check. CPU `scale:1.10` is reported separately; it does not block (#124).

### Timeouts

Every `fzb compile` / `inspect` / `run` is bounded by the model's
`time_budget_s` (manifest). The inner command is wrapped in GNU
`timeout --signal=TERM --kill-after=5`; the container is named
`fzb-bench-*` so a host-side wall-clock expiry can `docker stop` it.
A timeout is recorded as **`status: timeout`** with elapsed time and
whatever trials/estimate were produced — it is **not a z-test failure**
and rows are not dropped. Overall `timeout` is visible and is not an
accuracy pass. In A/B, **head timeout / base ok is a regression
(`block`)**; both sides timing out is `timeout`.

### Hosts

| target | host (docker context) | device | time? |
|---|---|---|---|
| `v100s` | xerxes `--gpus device=0` | Tesla V100S | yes; take `/tmp/gpu-lock-xerxes-0` |
| `rtx3070` | xerxes `--gpus device=1` | RTX 3070 | yes; take `/tmp/gpu-lock-xerxes-1` |
| `cpu` | gateway | i7-9700, 8 cores | yes |
| — | xerxes CPU | noisy (Tdarr/Plex) | **no** |
| — | shelly, bubs | CI (`fuzzybits-ci`) | **no** |

GPU jobs also use those devices: check `nvidia-smi --query-compute-apps` and
retry if busy. Do not time GPU work without the lock.

### Injected regression (acceptance)

`--inject-regression scale:1.10` multiplies **head** trial counts by 1.10
and rounds up to a 2^20 pass. V100S must **block** (`ci_lo > 1.03` on
`gpu_total_ms`). On chinese the GPU pass is 32·2^20, so the 1.10× request
runs 1.33× work (see table above). `--inject-regression repeat` still runs
`fzb run` twice under one profiler session on head (≈2× GPU time).
`--inject-regression sleep:MS` sleeps on head *outside* the sampled region, so
it no longer moves `cpu_sample_s`.

CPU `scale:1.10` numbers are reported; they do not block (#124).

### generic-pwr-model compatibility shim

The converted OpenPSA XMLs (submodule `ce2ef006…`) drop some gate / basic-event
definitions into sibling files: `data/saphsolve/*.JSInp` and
`data/openpsa/be_name_label_and_value_V1.2.csv` (BE/G/S/INIT/FE/FT prefixes;
`BE0` is the single-child placeholder). That is not MEF-compliant. v1 does
**not** patch the submodule. `benchmarks/datasets/compat/generic_pwr.py`
deterministically rewrites a cached copy, keyed by source hashes and shim
version, and only for `source: pwr`:

| model | exporter gap | shim |
|---|---|---|
| `EQK-BIN4/Grp-5` | `G3` missing in FT42/FT44; `G10` missing in FT51 (not defined in any of the 95 XMLs; present in the matching JSInp as `or` of G15/G19/G24/G27/G39/G40) | inject those `define-gate`s into the referencing fault tree |
| `EQK-BIN4/Grp-6` | `G3` is private in FT3; G300 in other FTs references it without a path. After cloning G3, MEF still rejects cyclic `FE56`/`FE57` fork nesting (`FE56` then `FE57` on some paths, the reverse on others). | clone `G3`; **known failure** `validity_error` — not reconstructable without rewriting the event tree |
| `EQK-BIN5/Grp-2` | fork `functional-event="F44"`; defined name is `FE44`. Same G3 clone gap | rewrite `F44`→`FE44`; clone `G3` |

The engine MEF loader stays strict. `fzb compile` now prints
`MEF initialize: Undefined Element Error; reference=…; base_path=…` from
`boost::errinfo` so a future gap names the missing element. There is no
engine-side opt-in to accept undefined names. Private cross-FT refs that
already have a per-FT `define-gate` clone are left alone. Manifest rows
record `compat: {shim: generic_pwr, version: 1}`. If a file still cannot be
reconstructed it stays as a known compile failure with a reason. Shim v1
compiles Grp-5 and Grp-2; Grp-6 remains `compile.expected = validity_error`
because FE56/FE57 fork order is cyclic.

This shim is temporary. [#125](https://github.com/fuzzybit-org/fuzzybits/issues/125)
(SAPHSOLVE `.JSInp` import: recover generic PWR models, general parser) will
read the sibling JSInp files directly. Delete
`benchmarks/datasets/compat/generic_pwr.py` once #125 recovers those models.
