# Python package (`fuzzybits`)

The Monte Carlo engine is installable as **`fuzzybits`** (import name and
distribution name). Wheels are Linux x86_64, self-contained: they bundle the
AdaptiveCpp runtime (`libacpp-rt`, `libacpp-common`, CUDA/OpenMP backend
plugins) and, for the `generic` variant, the SSCP JIT pieces (`llvm-to-backend`,
`libLLVM`, `libdevice.10.bc`, `opt`/`llc`). NVIDIA **driver** libraries
(`libcuda.so*`) are not bundled; a CUDA ≥ 12.8 driver is required for GPU
runs. Wheels are built with `MALLOC_TYPE=malloc` — no `LD_PRELOAD`.

Publishing to PyPI is [#36](https://github.com/fuzzybit-org/fuzzybits/issues/36)
and is gated on a legal review. Until then, install a locally built wheel.

## Install a wheel

```bash
pip install dist/fuzzybits-*.whl
python -c "import fuzzybits; print(fuzzybits.__version__, fuzzybits.list_devices())"
```

No `PYTHONPATH` or `LD_PRELOAD`. `import mcscram_native` still works (shim
that re-exports the active native extension) for the bench server.

Variants follow [#29](https://github.com/fuzzybit-org/fuzzybits/issues/29):

| wheel | native module | AdaptiveCpp flow | when to use |
|---|---|---|---|
| `generic` (default) | `fuzzybits._native_generic` | SSCP JIT | sm_86 / sm_120 / CPU |
| `cuda-sm70` | `fuzzybits._native_cuda_sm70` | clang CUDA SMCP AOT + embedded PTX | Volta (V100) |

Metal is out of scope here ([#30](https://github.com/fuzzybit-org/fuzzybits/issues/30)).

Import selection (override with `fuzzybits.init(variant=...)` or
`FUZZYBITS_VARIANT`): **cuda-sm70** if that extension is installed *and* an
sm_70 device is present, otherwise **generic**.

This phase ships one variant per wheel under the same distribution name, so
pip will replace one with the other. The import machinery still auto-selects
if both `.so` files are present (future `fuzzybits-native-*` split).

## API

```python
import fuzzybits

fuzzybits.init(backend="auto")          # "cuda" | "omp" | "auto"; also FUZZYBITS_BACKEND
print(fuzzybits.__version__)
print(fuzzybits.list_devices())

result = fuzzybits.probability(
    "model.xml",
    trials=1e7,
    pdag_c=2,
    no_kn=True,
    policy="wald",
)
print(result.estimate, result.ci, result.std_err)

pcfb = fuzzybits.compile("model.xml", pdag_c=2, no_kn=True)   # -> bytes
summary = fuzzybits.inspect(pcfb)  # fzb inspect --json keys + diagnostics
print(summary["queries"], summary["diagnostics"])

rows = fuzzybits.run(pcfb, trials=1e6, backend="auto")        # list[QueryResult]
print(rows[0].name, rows[0].estimate, rows[0].hits, rows[0].n)

with fuzzybits.Session(pcfb, backend="auto") as session:
    print(session.sample_shape)
    graph = session.graph()              # reconstructed PDAG, same keys as Pdag.to_graph()
    plan = session.execution_plan()      # same keys as Pdag.to_execution_plan()
    for obs in session.passes(max_passes=8):
        tally = obs.nodes[obs.target_index]
        print(obs.pass_index, tally.num_one_bits, tally.total_bits)
        if tally.total_bits >= 1_000_000:
            break                        # early stop; Session.close() on __exit__
```

`compile` / `compile_file` lower OpenPSA/MEF to a pcfb 1.1 buffer (the same
host path as `fzb compile`). `inspect` always runs `pcfb::validate` and
returns its diagnostics (empty iff the circuit is core-1.1 conforming).
`run` is the one-shot `fzb run` equivalent: `list[QueryResult(name, estimate,
ci_low, ci_high, n, hits)]`. `Session` reconstructs the PDAG and streams
raw per-pass popcounts (`num_one_bits` / `total_bits`, pass index, trials
per pass) so a host-side convergence loop can replace
`run_compile` + `layer_manager.single_pass_all` without changing results.

The wheel now links `fuzzybits_compile`, pcfb, and the runtime into the
native extension (`libmc` already carries `run.cc`). `fzb` / `scram-cli`
are not installed into the wheel.

Results are deterministic for a given model and trial count (Philox
counters). `seed=` raises `NotImplementedError` on `probability`, `compile`,
`run`, and `Session` ([#65](https://github.com/fuzzybit-org/fuzzybits/issues/65)).

`init()` must run before AdaptiveCpp initialises. The first call to
`list_devices` / `probability` / `compile` / `run` / `Session` /
`import mcscram_native` performs a lazy `init("auto")`. The JIT kernel
cache defaults to `~/.cache/fuzzybits/acpp` (`ACPP_APPDB_DIR`); set that
variable yourself to override.

## Build wheels from source

Requires the pinned toolchain in [`toolchain.env`](../toolchain.env)
(AdaptiveCpp v25.10.0, clang 20, CUDA 12.8.1). CMake is driven by
**scikit-build-core** using the same cache variables as
[`CMakePresets.json`](../CMakePresets.json) (`generic` by default).

```bash
pip install . --config-settings=cmake.define.ACPP_TARGETS=generic
# or the Volta AOT variant:
pip install . --config-settings=cmake.define.ACPP_TARGETS=cuda:sm_70 \
              --config-settings=cmake.define.ACPP_USE_ACCELERATED_CPU=off \
              --config-settings=cmake.define.ACPP_STDPAR=off \
              --config-settings=cmake.define.ACPP_EXTRA_COMPILE_OPTIONS=--cuda-include-ptx=sm_70
```

Containerised build (remote daemon, writes `dist/` on the client):

```bash
# 1. AdaptiveCpp+clang toolchain (full profile — SSCP)
docker --context xerxes build -f Dockerfile.cuda12.8 \
  --build-arg ACPP_COMPILER_FEATURE_PROFILE=full \
  --target adaptivecpp-clang \
  -t fuzzybits-acpp:full .

# 2. generic + cuda-sm70 wheels
docker --context xerxes build -f docker/wheel.Dockerfile \
  --build-arg TOOLCHAIN_IMAGE=fuzzybits-acpp:full \
  --build-arg FUZZYBITS_VARIANT=generic \
  --output type=local,dest=dist/generic .

docker --context xerxes build -f docker/wheel.Dockerfile \
  --build-arg TOOLCHAIN_IMAGE=fuzzybits-acpp:full \
  --build-arg FUZZYBITS_VARIANT=cuda-sm70 \
  --output type=local,dest=dist/cuda-sm70 .
```

`auditwheel repair` on Ubuntu 24.04 tags **`manylinux_2_39`**. A
`manylinux_2_28` rebuild is a follow-up. Driver libs are passed to
`--exclude libcuda.so.1`. Approximate wheel sizes (cp312, x86_64):
`generic` and `cuda-sm70` are both ~66 MB, dominated by `libLLVM.so.20.1`
(~153 MB uncompressed) for SSCP JIT.

## Tests

```bash
pip install pytest
pytest tests/python -q -m "not gpu"     # OpenMP, small trial counts
pytest tests/python/test_pcfb_gpu.py -q # CUDA bit-identity (separate interpreter)
```
