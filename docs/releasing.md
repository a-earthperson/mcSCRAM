# Releasing to PyPI

[`.github/workflows/release.yml`](../.github/workflows/release.yml) builds and publishes two PyPI projects:

- `fuzzybits`: the native engine wheels (Linux x86_64 manylinux, macOS arm64, macOS x86_64; CPython 3.12–3.15). Windows x86_64 is opt-in via `RELEASE_INCLUDE_WINDOWS`.
- `fuzzybits-server`: the FastAPI server with the bundled web UI (`py3-none-any`), pinned to the same `fuzzybits` version. It is built once `tools/server` is the `fuzzybits-server` distribution (#86).

Publishing uses PyPI trusted publishing (OIDC), so there are no API tokens or secrets.

## One-time setup

1. On https://pypi.org (Account → Publishing → Add a new pending publisher), add one GitHub publisher per project:

   | Field | `fuzzybits` | `fuzzybits-server` |
   |---|---|---|
   | PyPI Project Name | `fuzzybits` | `fuzzybits-server` |
   | Owner | `fuzzybit-org` | `fuzzybit-org` |
   | Repository name | `fuzzybits` | `fuzzybits` |
   | Workflow name | `release.yml` | `release.yml` |
   | Environment name | `pypi` | `pypi-server` |

   The environments must differ: PyPI allows one pending publisher per (repository, workflow, environment). A pending publisher does not reserve the name; the first upload creates the project.
2. In the GitHub repository settings, create environments `pypi` and `pypi-server`: required reviewer (a maintainer); deployment rule "selected tags" `v*`.

   The server is uploaded only after the engine upload for the same version succeeds.
3. Optional repository variables:
   - `RELEASE_INCLUDE_WINDOWS=true` builds the Windows wheel in releases. It is off by default until Windows CI is reliable (#108, #109).
   - `PYPI_MAX_FILE_MB`: set this once PyPI raises the per-file limit (see Limits).

## Dry run

Actions → Release → Run workflow (on any branch). Every version string becomes `X.Y.Z.dev<run number>`, and all wheels are built and smoke-tested (including `pip install` of the engine and server wheels). The `wheels-report` job lists every artefact, applies the per-file size gate, and prints the total MB against PyPI's 10 GB project cap. Nothing is uploaded; the wheels are kept as workflow artifacts for 14 days. CPython 3.12–3.15 × Linux x86_64, macOS arm64, and macOS x86_64.

## Cut a release

1. From a clean `develop` that matches `origin/develop`, run:
   ```bash
   scripts/release.sh patch          # or minor, major, or an explicit 0.19.0
   scripts/release.sh --dry-run minor  # rewrite and show the diff only
   ```
   The script sets the version in every file that carries it:
   - `pyproject.toml` and `python/fuzzybits/__init__.py`
   - `tools/server/pyproject.toml` (`version` and the `fuzzybits==` pin)
   - `project(SCRAM VERSION …)` in `CMakeLists.txt`
   - the Docker `ARG`s, the `.env` server tag, the README examples, and both `uv.lock` files

   It fails if any of those files no longer matches its pattern, and it lists any other mention of the old version without changing it. It then commits `chore(release): X.Y.Z` to `develop`, creates the annotated tag `vX.Y.Z`, and, after asking, pushes both in one atomic push. `--no-push` stops after the local commit and tag; `-y` pushes without asking. The `version` job in `release.yml` fails if the tag and any version string disagree.
2. Approve the `pypi` and then the `pypi-server` deployment. After both uploads, `github-release` creates the GitHub Release with every wheel attached.

The tag also redeploys live.fuzzybit.org (`server-image.yml`), and the `develop` push redeploys dev.fuzzybit.org.

If an upload fails partway, fix the cause and rerun the failed jobs (`gh run rerun <run id> --failed`): the uploads use `skip-existing`, so files already on PyPI are skipped. PyPI never accepts changed files for an existing version; to fix a bad release, publish a new version.

## Limits

- **File size:** PyPI rejects files over 100 MB by default. The Windows wheel is about 270 MB, so the publish jobs hold back any file over `PYPI_MAX_FILE_MB` (default 100) with a warning. Held-back wheels are attached only to the GitHub Release. Request a higher limit for `fuzzybits` at https://github.com/pypi/support, then set `PYPI_MAX_FILE_MB`. The 10 GB per-project total will also need raising after roughly 30 releases at this size.
- **cuda-sm70:** not published. It is the same `fuzzybits` distribution with the same Linux platform tag as the generic wheel, so PyPI can hold only one of them.
- **Python:** wheels are CPython 3.12, 3.13, 3.14 and 3.15 (abi3/stable ABI is not used; pybind11 3.x, no free-threaded `t` builds). On other Python versions pip finds no `fuzzybits` wheel. No sdist is published, because building from source needs the AdaptiveCpp toolchain.
- **Platforms:** the Linux wheels are built on Ubuntu 24.04 and carry the manylinux tag auditwheel assigns for that glibc (currently `manylinux_2_39`), which may exclude older distributions. macOS arm64 wheels need macOS 26+ on Apple Silicon (Metal + OpenMP). macOS x86_64 wheels are built on GitHub-hosted `macos-15-intel` with deployment target 15.0, so they install on Intel Macs running macOS 15+, including 26; they are OpenMP CPU only (no Metal).
