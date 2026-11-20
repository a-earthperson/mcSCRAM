from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

# Optional data mount from fuzzybits-bench (CSVs, OpenPSA/Aralia XML).
# Historical run tables stay in that private repo; this tree never ships them.
SERVER_DATA_DIR_ENV = "FUZZYBITS_SERVER_DATA_DIR"
SERVER_ENGINE_ENV = "FUZZYBITS_SERVER_ENGINE"
SERVER_CACHE_DIR_ENV = "FUZZYBITS_SERVER_CACHE_DIR"
SERVER_UI_ENV = "FUZZYBITS_SERVER_UI"
SERVER_UI_DIR_ENV = "FUZZYBITS_SERVER_UI_DIR"
SERVER_BENCH_DATA_URL_ENV = "FUZZYBITS_BENCH_DATA_URL"
SERVER_BENCH_CACHE_DIR_ENV = "FUZZYBITS_BENCH_CACHE_DIR"
DEFAULT_BENCH_INDEX_URL = "https://fuzzybit-org.github.io/fuzzybits/bench/v1/index.json"
_LEGACY_DEFAULT_INPUT = Path("/app/input/SmallTree.xml")
_BOOLEAN_TRUE = {"1", "true", "yes", "on"}
_BOOLEAN_FALSE = {"0", "false", "no", "off"}
_BACKENDS = ("auto", "cuda", "omp", "metal")
_ENGINES = ("pcfb", "legacy")


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _BOOLEAN_TRUE


def server_engine() -> str:
    """Compile/sample engine: ``pcfb`` (default) or ``legacy`` in-memory PDAG."""
    raw = os.getenv(SERVER_ENGINE_ENV, "pcfb").strip().lower()
    if raw in {"legacy", "memory", "in-memory", "run_compile"}:
        return "legacy"
    return "pcfb"


def parse_watch_value(raw: Any) -> tuple[bool | None, list[str]]:
    """Split ``watch`` into (watch_mode, names).

    ``MCSCRAM_WATCH=1`` is the C++ progress-bar flag. Any other non-boolean
    string is a comma-separated MEF name / PDAG-index list, matching how
    :mod:`fuzzybits_server.convergence.sampling` registers extra tally nodes.
    """
    if raw is None or raw is False:
        return None, []
    if raw is True:
        return True, []
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return None, [str(int(raw))]
    if isinstance(raw, (list, tuple, set)):
        return None, [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    if not text:
        return None, []
    lowered = text.lower()
    if lowered in _BOOLEAN_TRUE:
        return True, []
    if lowered in _BOOLEAN_FALSE:
        return False, []
    return None, [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def default_cache_dir() -> Path:
    """Per-user cache root (platformdirs); used when ``FUZZYBITS_SERVER_CACHE_DIR`` is unset."""
    from platformdirs import user_cache_path

    return user_cache_path("fuzzybits", "fuzzybit-org") / "server"


def pcfb_cache_dir() -> Path:
    raw = os.getenv(SERVER_CACHE_DIR_ENV, "").strip()
    path = Path(raw).expanduser() if raw else default_cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def server_data_dir() -> Path | None:
    """Directory of optional bench data (CSVs, model XML).

    Set ``FUZZYBITS_SERVER_DATA_DIR`` to a fuzzybits-bench checkout or mount.
    Returns ``None`` when unset or not a directory — the server still starts;
    model listing is empty and default input files are omitted.
    """
    raw = os.getenv(SERVER_DATA_DIR_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def optional_table_path(name: str) -> Path | None:
    """Locate ``compiler.csv`` / ``convergence.csv`` under the data dir.

    These tables are not imported into this public repo. fuzzybits-bench can
    mount its copies via ``FUZZYBITS_SERVER_DATA_DIR``.
    """
    data = server_data_dir()
    if data is None:
        return None
    path = data / name
    return path if path.is_file() else None


def input_files_from_env() -> list[str]:
    raw = os.getenv("MCSCRAM_INPUT_FILES")
    if raw is not None:
        return [part.strip() for part in raw.replace(":", ",").split(",") if part.strip()]

    data = server_data_dir()
    if data is not None:
        for candidate in (
            data / "SmallTree.xml",
            data / "input" / "SmallTree.xml",
            data / "input" / "SmallTree" / "SmallTree.xml",
        ):
            if candidate.is_file():
                return [str(candidate)]

    if _LEGACY_DEFAULT_INPUT.is_file():
        return [str(_LEGACY_DEFAULT_INPUT)]
    return []


def options_from_env() -> dict[str, Any]:
    options: dict[str, Any] = {}

    for env_name, key in (
        ("MCSCRAM_NUM_TRIALS", "num_trials"),
        ("MCSCRAM_REQUESTED_TRIALS_PER_PASS", "requested_trials_per_pass"),
        ("MCSCRAM_DELTA", "delta"),
        ("MCSCRAM_BURN_IN", "burn_in"),
        ("MCSCRAM_CONFIDENCE", "confidence"),
        ("MCSCRAM_ORACLE", "oracle"),
        ("MCSCRAM_MISSION_TIME", "mission_time"),
        ("MCSCRAM_TIME_STEP", "time_step"),
        ("MCSCRAM_CUT_OFF", "cut_off"),
    ):
        if raw := os.getenv(env_name):
            options[key] = float(raw)

    for env_name, key in (
        ("MCSCRAM_SEED", "seed"),
        ("MCSCRAM_LIMIT_ORDER", "limit_order"),
        ("MCSCRAM_COMPILATION_PASSES", "compilation_passes"),
        ("FUZZYBITS_PASSES_PER_STEP", "passes_per_step"),
    ):
        if raw := os.getenv(env_name):
            options[key] = int(raw)

    for env_name, key in (
        ("MCSCRAM_EARLY_STOP", "early_stop"),
        ("MCSCRAM_NO_KN", "no_kn"),
        ("MCSCRAM_NO_XOR", "no_xor"),
        ("MCSCRAM_RECOGNIZE_ATLEAST_GROUPS", "recognize_atleast_groups"),
        ("MCSCRAM_RECOGNIZE_XOR_GROUPS", "recognize_xor_groups"),
        ("MCSCRAM_PREPROCESSOR", "preprocessor"),
        ("MCSCRAM_PRINT", "print"),
        ("MCSCRAM_IMPORTANCE", "importance"),
        ("MCSCRAM_UNCERTAINTY", "uncertainty"),
        ("MCSCRAM_CCF", "ccf"),
        ("MCSCRAM_SIL", "sil"),
    ):
        if raw := os.getenv(env_name):
            options[key] = raw.strip().lower() in {"1", "true", "yes", "on"}

    if watch_raw := os.getenv("MCSCRAM_WATCH"):
        mode, names = parse_watch_value(watch_raw)
        if mode is not None:
            options["watch_mode"] = mode
        if names:
            options["watch"] = names
    if watch_nodes := os.getenv("MCSCRAM_WATCH_NODES"):
        _, names = parse_watch_value(watch_nodes)
        if names:
            existing = [str(item) for item in options.get("watch", [])] if isinstance(options.get("watch"), list) else []
            options["watch"] = existing + names

    if policy := os.getenv("MCSCRAM_POLICY"):
        options["policy"] = policy

    if overrides := os.getenv("MCSCRAM_OPTIONS_JSON"):
        parsed = json.loads(overrides)
        if not isinstance(parsed, dict):
            raise ValueError("MCSCRAM_OPTIONS_JSON must decode to a JSON object")
        options.update(parsed)

    return options


def parse_server_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fuzzybits-server",
        description="Serve the fuzzybits API and playground UI on one port.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    parser.add_argument("--data-dir", default=None, metavar="DIR", help="OpenPSA/Aralia XML root (FUZZYBITS_SERVER_DATA_DIR)")
    parser.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help=f"compiled .pcfb cache (default: {default_cache_dir()})",
    )
    parser.add_argument(
        "--backend",
        choices=_BACKENDS,
        default="auto",
        help="AdaptiveCpp visibility: auto (default), cuda, omp, or metal",
    )
    parser.add_argument(
        "--engine",
        choices=_ENGINES,
        default=None,
        help="compile/sample engine: pcfb (default) or legacy in-memory PDAG",
    )
    parser.add_argument("--no-ui", action="store_true", help="disable the playground UI; GET / returns an API pointer")
    parser.add_argument("--open", action="store_true", help="open the UI in a browser")
    return parser.parse_args(argv)


def apply_server_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags and apply them to the process environment."""
    args = parse_server_cli(argv)
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    if args.data_dir:
        os.environ[SERVER_DATA_DIR_ENV] = str(Path(args.data_dir).expanduser())
    if args.cache_dir:
        os.environ[SERVER_CACHE_DIR_ENV] = str(Path(args.cache_dir).expanduser())
    if args.backend != "auto":
        os.environ["ACPP_VISIBILITY_MASK"] = args.backend
        os.environ["FUZZYBITS_BACKEND"] = args.backend
    if args.engine:
        os.environ[SERVER_ENGINE_ENV] = args.engine
    if args.no_ui:
        os.environ[SERVER_UI_ENV] = "0"
    return args


def server_ui_enabled() -> bool:
    return env_bool(SERVER_UI_ENV, True)


def _ui_index(path: Path) -> Path | None:
    index = path / "index.html"
    return path if path.is_dir() and index.is_file() else None


def packaged_ui_dir() -> Path | None:
    """CSR build shipped in the wheel at ``fuzzybits_server/webui/``."""
    try:
        traversable = files("fuzzybits_server") / "webui"
    except (ModuleNotFoundError, FileNotFoundError, TypeError, AttributeError, NotADirectoryError):
        return None
    try:
        if not traversable.is_dir() or not (traversable / "index.html").is_file():
            return None
    except (FileNotFoundError, OSError, TypeError, NotADirectoryError):
        return None
    path = Path(traversable) if isinstance(traversable, Path) else Path(str(traversable))
    return _ui_index(path)


def source_ui_dir() -> Path | None:
    """``tools/server/webui/dist`` next to a src checkout or the Docker image copy."""
    return _ui_index(Path(__file__).resolve().parents[2] / "webui" / "dist")


def server_ui_dir() -> Path | None:
    """Directory of the CSR playground build (``index.html`` + ``assets/``).

    ``FUZZYBITS_SERVER_UI_DIR`` wins. Otherwise the copy bundled in the wheel
    (``importlib.resources``), then ``tools/server/webui/dist`` for a source
    checkout or the server image.
    """
    raw = os.getenv(SERVER_UI_DIR_ENV, "").strip()
    if raw:
        return _ui_index(Path(raw).expanduser())
    return packaged_ui_dir() or source_ui_dir()


def bench_index_url() -> str:
    raw = os.getenv(SERVER_BENCH_DATA_URL_ENV, "").strip()
    return raw or DEFAULT_BENCH_INDEX_URL


def bench_cache_dir() -> Path | None:
    raw = os.getenv(SERVER_BENCH_CACHE_DIR_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


@dataclass(frozen=True)
class CalculationConfig:
    input_files: list[str]
    options: dict[str, Any]
    pcfb_path: str | None = None

    @classmethod
    def from_env(cls) -> CalculationConfig:
        return cls(input_files=input_files_from_env(), options=options_from_env())

    def with_options(self, **overrides: Any) -> CalculationConfig:
        if not overrides:
            return self
        options = dict(self.options)
        options.update(overrides)
        return CalculationConfig(
            input_files=self.input_files,
            options=options,
            pcfb_path=self.pcfb_path,
        )
