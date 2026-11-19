# Releasing to PyPI

[`.github/workflows/release.yml`](../.github/workflows/release.yml) builds and publishes two PyPI projects:

- `fuzzybits`: the native engine wheels (Linux x86_64 manylinux, macOS arm64, Windows x86_64; CPython 3.12).
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

   Repeat on https://test.pypi.org (separate account) with environments `testpypi` and `testpypi-server`. The environments must differ: PyPI allows one pending publisher per (repository, workflow, environment). A pending publisher does not reserve the name; the first upload creates the project.
2. In the GitHub repository settings, create environments:
   - `pypi` and `pypi-server`: required reviewer (a maintainer); deployment rule "selected tags" `v*`.
   - `testpypi` and `testpypi-server`: no reviewer.

   The server is uploaded only after the engine upload for the same version succeeds.
3. Optional: once PyPI raises the per-file limit, set the repository variable `PYPI_MAX_FILE_MB` (see Limits).

## Rehearse on TestPyPI

Actions → Release → Run workflow (on `develop`). Every version string becomes `X.Y.Z.dev<run number>`, all wheels are built and smoke-tested, and the result is uploaded to TestPyPI. TestPyPI does not carry the server's dependencies, so install with both indexes:

```bash
pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ fuzzybits-server
```

## Cut a release

1. Set the version everywhere and commit it to `develop` (via a PR):
   ```bash
   scripts/ci/release-version.sh apply 0.18.0
   (cd tools/server && uv lock)
   ```
   This updates `pyproject.toml`, `python/fuzzybits/__init__.py` and, for the server, `tools/server/pyproject.toml` (`version` and the `fuzzybits==` pin). `uv lock` refreshes `tools/server/uv.lock`, which records both versions.
2. Tag that commit and push the tag:
   ```bash
   git tag -a v0.18.0 -m "fuzzybits 0.18.0" && git push origin v0.18.0
   ```
   The `version` job fails if the tag and any version string disagree.
3. Approve the `pypi` deployment. After the upload, `github-release` creates the GitHub Release with every wheel attached.

PyPI never accepts the same version twice; to fix a bad release, publish a new version.

## Limits

- **File size:** PyPI rejects files over 100 MB by default. The Windows wheel is about 270 MB, so the publish jobs hold back any file over `PYPI_MAX_FILE_MB` (default 100) with a warning. Held-back wheels are attached only to the GitHub Release. Request a higher limit for `fuzzybits` at https://github.com/pypi/support, then set `PYPI_MAX_FILE_MB`. The 10 GB per-project total will also need raising after roughly 30 releases at this size.
- **cuda-sm70:** not published. It is the same `fuzzybits` distribution with the same Linux platform tag as the generic wheel, so PyPI can hold only one of them.
- **Python:** wheels are CPython 3.12 only. On other Python versions pip finds no `fuzzybits` wheel. No sdist is published, because building from source needs the AdaptiveCpp toolchain.
- **Platforms:** the Linux wheel is built on Ubuntu 24.04 and carries the manylinux tag auditwheel assigns for that glibc, which may exclude older distributions. The macOS wheel needs macOS 26 on Apple Silicon.
