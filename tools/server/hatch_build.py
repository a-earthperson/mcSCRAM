"""Hatchling hook: bundle the CSR UI into wheel/sdist; skip editable installs."""

from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        # `uv sync` uses the editable target and must not require Node / webui/dist.
        if version == "editable":
            return
        dist = Path(self.root) / "webui" / "dist"
        index = dist / "index.html"
        if not index.is_file():
            raise RuntimeError(
                "webui/dist/index.html is missing. Build the CSR UI before "
                "packaging: (cd tools/server/webui && npm ci && npm run build) "
                "&& (cd tools/server && uv build --wheel --out-dir ../../dist-server)"
            )
        for junk in dist.rglob("._*"):
            junk.unlink()
        for junk in dist.rglob(".DS_Store"):
            junk.unlink()
        force_include = build_data.setdefault("force_include", {})
        if self.target_name == "wheel":
            force_include[str(dist)] = "fuzzybits_server/webui"
            build_data["pure_python"] = True
        elif self.target_name == "sdist":
            force_include[str(dist)] = "webui/dist"
