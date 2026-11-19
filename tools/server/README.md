<!--
SPDX-FileCopyrightText: 2026 fuzzybit-org
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# tools/server — FastAPI Monte Carlo service

AGPL-3.0-or-later. Imported from `fuzzybit-org/fuzzybits-bench` (`src/server`)
with history. Compile products, WebSocket sampling, and Wald/Bayes/GBAS
convergence. Default engine compiles OpenPSA/MEF to `.pcfb` once per
`(source SHA-256s, compile options)` and samples through
`fuzzybits.Session`. Set `FUZZYBITS_SERVER_ENGINE=legacy` for the previous
in-memory `run_compile` path. The `MCSCRAM_CONVERGENCE_ENGINE=native`
`run_probability` fallback is unchanged.

## Run

```bash
cd tools/server
uv sync --extra dev
uv run mcscram-server          # HOST=0.0.0.0 PORT=8000
```

Compile / sampling endpoints need the native module. Install a `fuzzybits`
wheel built from this repo (not on PyPI; [#36](https://github.com/fuzzybit-org/fuzzybits/issues/36)):

```bash
uv pip install ../../dist/fuzzybits-*.whl
python -c "import mcscram_native; print(mcscram_native)"
```

Without the wheel the process still starts: `/api/v1/healthz` works, model
listing is empty, and native compile returns an install error instead of
aborting the interpreter.

## Test

The imported pytest suite is pure Python (BernoulliSampler, numpy, scipy)
plus config/cache helpers. It does not import `mcscram_native` unless a
test is marked `@pytest.mark.native`.

```bash
cd tools/server
uv sync --frozen --extra dev
uv run pytest -m "not native"
```

Mark future engine tests `@pytest.mark.native`. `tests/conftest.py` skips
them when the wheel is absent.

## Data (Aralia XML + optional bench CSVs)

Public OpenPSA/Aralia models live in this repo at `input/Aralia/*.xml`.
Historical `compiler.csv` / `convergence.csv` stay in fuzzybits-bench
(CC BY-SA 4.0) and are optional. Point the server at a data root:

```bash
export FUZZYBITS_SERVER_DATA_DIR=/path/to/data
# docker: mount this repo's models (CSVs optional)
#   -v $PWD/input/Aralia:/data/Aralia
#   -e FUZZYBITS_SERVER_DATA_DIR=/data
```

| Path under `$FUZZYBITS_SERVER_DATA_DIR` | Used for |
| --- | --- |
| `compiler.csv`, `convergence.csv` | Optional run tables (`optional_table_path`) |
| `*.xml`, `Aralia/`, `input/`, `input/Aralia/` | Model catalog (`GET /api/v1/model/xml-inputs`) |
| `SmallTree.xml` or `input/SmallTree/SmallTree.xml` | Default `MCSCRAM_INPUT_FILES` when that env is unset |

`MCSCRAM_ARALIA_INPUT_DIR` still overrides the XML search. If neither env is
set and `/app/input` is missing, the catalog is empty and the server stays up.

## Env

| Variable | Role |
| --- | --- |
| `FUZZYBITS_SERVER_DATA_DIR` | Bench data mount (CSVs + XML) |
| `FUZZYBITS_SERVER_ENGINE` | `pcfb` (default) or `legacy` in-memory PDAG |
| `FUZZYBITS_SERVER_CACHE_DIR` | Directory for compiled / uploaded `.pcfb` files |
| `MCSCRAM_ARALIA_INPUT_DIR` | Extra XML search root |
| `MCSCRAM_INPUT_FILES` | Explicit model path(s), comma/colon separated |
| `MCSCRAM_WATCH` | Boolean watch-mode **or** comma-separated MEF names / PDAG indices to tally |
| `MCSCRAM_WATCH_NODES` | Extra comma-separated watch names (same identification as `MCSCRAM_WATCH`) |
| `MCSCRAM_AUTOSTART` | Start sampling on boot (default false) |
| `HOST` / `PORT` / `LOG_LEVEL` | uvicorn |

Sampling knobs (`MCSCRAM_NUM_TRIALS`, `MCSCRAM_POLICY`, …) are unchanged.

## `.pcfb` upload

`POST /api/v1/model/pcfb` accepts a raw `.pcfb` body (`Content-Type:
application/octet-stream`). The server validates with `fuzzybits.inspect`
(`pcfb::validate` diagnostics). Invalid circuits return HTTP 400:

```json
{"detail": {"error": "...", "diagnostics": [...], "inspect": {...}}}
```

Success:

```json
{"id": "<sha256>", "pcfb_id": "<sha256>", "path": "...", "inspect": {...}}
```

Start a run on the uploaded circuit:

```bash
curl -sS -X POST "$HOST/api/v1/model/pcfb" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @model.pcfb
curl -sS -X POST "$HOST/api/v1/start" \
  -H "Content-Type: application/json" \
  -d '{"pcfb_id":"<sha256>","watch":["g1"]}'
```

`GET /api/v1/model/{dag,kernel-dag,sample-shape,execution-plan,dag-bundle}`
accept `?pcfb_id=<sha256>` to serve those views from the stored circuit
instead of compiling MEF.

## Image

`ghcr.io/fuzzybit-org/fuzzybits-server` is `FROM` the slim runtime image
(`ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-generic` by default;
`scram-cli`, `fzb`, AdaptiveCpp, pip-installed `fuzzybits`). uv installs
this package into a venv with `--system-site-packages` so the engine is not
rebuilt. Variants match the runtime: `generic` (default) and `cuda12.8-sm70`.

Tags (same scheme as `fuzzybits-runtime`): `ci-<sha12>-<variant>` on `ci/**`,
`nightly-<variant>` + `sha-<sha12>-<variant>` on `develop`, `vX.Y.Z-<variant>`
on release tags.

The image runs as non-root `coder`. Default env:

| Variable | Image default |
| --- | --- |
| `FUZZYBITS_SERVER_DATA_DIR` | `/data` |
| `FUZZYBITS_SERVER_CACHE_DIR` | `/home/coder/.cache/fuzzybits/server` |
| `ACPP_APPDB_DIR` | `/home/coder/.cache/fuzzybits/acpp` |
| `ACPP_VISIBILITY_MASK` | `omp` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` |

Volumes: `/data` (models) and `/home/coder/.cache/fuzzybits` (JIT + `.pcfb` cache).
`HEALTHCHECK` hits `/api/v1/healthz`.

### `docker run`

```bash
IMG=ghcr.io/fuzzybit-org/fuzzybits-server:nightly-generic

# CPU (OpenMP)
docker run --rm -p 8000:8000 \
  -e ACPP_VISIBILITY_MASK=omp \
  -v "$PWD/input/Aralia:/data/Aralia:ro" \
  -v fuzzybits-cache:/home/coder/.cache/fuzzybits \
  "$IMG"

# GPU (NVIDIA device N). Use the cuda12.8-sm70 tag on Volta (sm_70);
# generic SSCP covers other NVIDIA GPUs.
docker run --rm --gpus device=0 -p 8000:8000 \
  -e ACPP_VISIBILITY_MASK=cuda \
  -v "$PWD/input/Aralia:/data/Aralia:ro" \
  -v fuzzybits-cache:/home/coder/.cache/fuzzybits \
  ghcr.io/fuzzybit-org/fuzzybits-server:nightly-cuda12.8-sm70
```

xerxes mapping: `--gpus device=0` is the Tesla V100S, `device=1` is the RTX 3070.
Hosted CI smokes CPU only (`ACPP_VISIBILITY_MASK=omp`).

```bash
curl -sS http://127.0.0.1:8000/api/v1/healthz
curl -sS http://127.0.0.1:8000/api/v1/model/xml-inputs
curl -sS -X POST http://127.0.0.1:8000/api/v1/start \
  -H "Content-Type: application/json" \
  -d '{"model_id":"chinese","num_trials":1e7,"early_stop":false}'
```

E2E helper (compile chinese + das9201, check \|z\|≤4 vs BDD refs, POST a
`fzb compile` `.pcfb`):

```bash
tools/server/scripts/e2e-smoke.sh "$IMG"
```

### Compose

```yaml
services:
  server:
    image: ghcr.io/fuzzybit-org/fuzzybits-server:nightly-generic
    ports: ["8000:8000"]
    environment:
      ACPP_VISIBILITY_MASK: omp          # cuda on GPU hosts
      HOST: "0.0.0.0"
      PORT: "8000"
    volumes:
      - ../../input/Aralia:/data/Aralia:ro
      - fuzzybits-cache:/home/coder/.cache/fuzzybits
    # gpus: ["device=0"]                 # NVIDIA: also set ACPP_VISIBILITY_MASK=cuda
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/healthz')"]
      interval: 15s
      timeout: 5s
      retries: 3

volumes:
  fuzzybits-cache:
```
