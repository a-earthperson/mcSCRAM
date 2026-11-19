"""Optional cached copy of published benchmark history (#123).

The CSR tab fetches ``/api/v1/bench/v1/index.json`` same-origin, then the
GitHub Pages URL. This module serves a local cache (``FUZZYBITS_BENCH_CACHE_DIR``)
or proxies the Pages URL so Docker / pip / compose work when the browser
cannot reach Pages.
"""

from __future__ import annotations

import json
import posixpath
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response

from fuzzybits_server.config import bench_cache_dir, bench_index_url

_FETCH_TIMEOUT_S = 12.0


def _safe_rel(path: str) -> str:
    cleaned = posixpath.normpath(path.lstrip("/"))
    if cleaned in {".", ""} or cleaned.startswith(".."):
        raise HTTPException(status_code=404, detail="not found")
    return cleaned


def _index_base(index_url: str) -> str:
    if index_url.endswith("index.json"):
        return index_url[: -len("index.json")]
    if not index_url.endswith("/"):
        return index_url + "/"
    return index_url


def _under_root(root: Path, candidate: Path) -> Path | None:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved


def _cache_file(rel: str, *, for_write: bool = False) -> Path | None:
    root = bench_cache_dir()
    if root is None:
        return None
    rel = _safe_rel(rel)
    candidates = [
        root / rel,
        root / "bench" / "v1" / rel,
        root / "v1" / rel,
    ]
    if for_write:
        chosen = _under_root(root, candidates[0])
        return chosen
    for candidate in candidates:
        path = _under_root(root, candidate)
        if path is not None and path.is_file():
            return path
    return None


def _read_cache(rel: str) -> bytes | None:
    path = _cache_file(rel)
    if path is None:
        return None
    return path.read_bytes()


def _write_cache(rel: str, body: bytes) -> None:
    path = _cache_file(rel, for_write=True)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "fuzzybits-server"})
    try:
        with urlopen(request, timeout=_FETCH_TIMEOUT_S) as response:
            return response.read()
    except HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise HTTPException(status_code=503, detail=f"benchmark history unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="benchmark history timed out") from exc


def _json_or_raw(body: bytes) -> Response:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return Response(content=body, media_type="application/octet-stream")
    return JSONResponse(payload)


def load_index() -> dict[str, Any]:
    cached = _read_cache("index.json")
    if cached is not None:
        return json.loads(cached.decode("utf-8"))
    body = _fetch(bench_index_url())
    _write_cache("index.json", body)
    return json.loads(body.decode("utf-8"))


def load_result(rel: str) -> bytes:
    rel = _safe_rel(rel)
    cached = _read_cache(rel)
    if cached is not None:
        return cached
    url = urljoin(_index_base(bench_index_url()), rel)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=404, detail="not found")
    body = _fetch(url)
    _write_cache(rel, body)
    return body


def create_bench_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/bench")

    @router.get("/config")
    def config() -> dict[str, Any]:
        cache = bench_cache_dir()
        return {
            "index_url": bench_index_url(),
            "same_origin": "/api/v1/bench/v1/index.json",
            "cache_dir": str(cache) if cache else None,
        }

    @router.get("/v1/index.json")
    def index() -> JSONResponse:
        try:
            payload = load_index()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=502, detail="history index is not JSON") from exc
        return JSONResponse(payload)

    @router.get("/v1/{rel:path}")
    def result(rel: str) -> Response:
        return _json_or_raw(load_result(rel))

    return router
