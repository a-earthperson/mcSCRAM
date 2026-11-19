"""CSR UI at ``/``, next to the API.

Enabled by default. Disable with ``--no-ui`` / ``FUZZYBITS_SERVER_UI=0``.
Static assets live in ``FUZZYBITS_SERVER_UI_DIR`` (default:
``tools/server/webui/dist`` in a source checkout or image).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from mcscram_server.config import server_ui_dir, server_ui_enabled

_INDEX_CACHE = {"Cache-Control": "no-cache"}
_ASSET_CACHE = "public, max-age=31536000, immutable"
_API_POINTER = {
    "service": "fuzzybits-server",
    "api": "/api/v1",
    "ui": False,
}


class ImmutableStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = _ASSET_CACHE
        return response


def _pointer(**extra: object) -> JSONResponse:
    payload = dict(_API_POINTER)
    payload.update(extra)
    return JSONResponse(payload)


def mount_ui(app: FastAPI) -> None:
    if not server_ui_enabled():
        @app.get("/")
        def ui_disabled() -> JSONResponse:
            return _pointer()

        return

    dist = server_ui_dir()
    if dist is None:
        @app.get("/")
        def ui_missing() -> JSONResponse:
            return _pointer(detail="UI assets not found; set FUZZYBITS_SERVER_UI_DIR or build tools/server/webui")

        return

    index = dist / "index.html"
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", ImmutableStaticFiles(directory=assets), name="ui-assets")

    def _index() -> FileResponse:
        return FileResponse(index, media_type="text/html", headers=_INDEX_CACHE)

    @app.get("/")
    def ui_index() -> FileResponse:
        return _index()

    @app.get("/{full_path:path}")
    def ui_spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/") or full_path in {"docs", "redoc", "openapi.json"}:
            raise HTTPException(status_code=404)
        candidate = (dist / full_path).resolve()
        try:
            candidate.relative_to(dist.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=404) from exc
        if candidate.is_file():
            return FileResponse(candidate)
        return _index()
