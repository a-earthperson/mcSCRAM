"""Backend selection, AdaptiveCpp env, and native-module loading.

AdaptiveCpp reads ``ACPP_VISIBILITY_MASK`` and the kernel-cache location at
runtime *initialisation*. Those variables must be set before the extension is
imported. Call :func:`init` (or rely on the lazy default) before any other API.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Literal

Backend = Literal["auto", "cuda", "omp"]
Variant = Literal["auto", "generic", "cuda-sm70"]

_NATIVE: ModuleType | None = None
_INITIALIZED = False
_BACKEND: str | None = None
_VARIANT: str | None = None
_DEVICE: str | None = None

_GENERIC_MOD = "_native_generic"
_CUDA_SM70_MOD = "_native_cuda_sm70"


def _cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return base / "fuzzybits" / "acpp"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "fuzzybits" / "acpp"


def _windows_dll_dirs() -> list[Path]:
    pkg = Path(__file__).resolve().parent
    return [
        pkg,
        pkg / "lib",
        pkg / "lib" / "hipSYCL",
        pkg / "lib" / "hipSYCL" / "llvm-to-backend",
        pkg / "lib" / "hipSYCL" / "ext" / "llvm" / "bin",
        pkg / ".libs",
    ]


def _ensure_windows_dll_search() -> None:
    """Put AdaptiveCpp DLLs on the Windows loader path before the .pyd imports.

    Python 3.8+ does not search PATH for extension DLL deps; AdaptiveCpp also
    spawns opt/llc/llvm-to-*-tool as child processes, which *do* search PATH.
    """
    if sys.platform != "win32":
        return
    dirs = [p for p in _windows_dll_dirs() if p.is_dir()]
    add = getattr(os, "add_dll_directory", None)
    for d in dirs:
        if add is not None:
            add(str(d))
    extra = os.pathsep.join(str(d) for d in dirs)
    if extra:
        os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")


def _ensure_jit_cache() -> None:
    """Point AdaptiveCpp's appdb/JIT cache at a user dir unless already set."""
    if os.environ.get("ACPP_APPDB_DIR"):
        return
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["ACPP_APPDB_DIR"] = str(cache)


def _ensure_runtime_defaults() -> None:
    os.environ.setdefault("ACPP_DEBUG_LEVEL", "0")
    os.environ.setdefault("ACPP_PERSISTENT_RUNTIME", "1")


def _load_cuda_driver():
    """Load the NVIDIA driver library. Intentionally not bundled in the wheel."""
    names = ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _cuda_compute_capabilities() -> list[tuple[int, int, int]]:
    """Return ``(ordinal, major, minor)`` for each CUDA device via libcuda.

    Does not initialise AdaptiveCpp. ``libcuda.so.1`` / ``nvcuda.dll`` is the
    NVIDIA *driver* library and is intentionally not bundled in the wheel.
    """
    cuda = _load_cuda_driver()
    if cuda is None:
        return []

    if cuda.cuInit(0) != 0:
        return []

    count = ctypes.c_int(0)
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value <= 0:
        return []

    out: list[tuple[int, int, int]] = []
    for ordinal in range(count.value):
        dev = ctypes.c_int(0)
        if cuda.cuDeviceGet(ctypes.byref(dev), ordinal) != 0:
            continue
        major = ctypes.c_int(0)
        minor = ctypes.c_int(0)
        if cuda.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), dev) != 0:
            continue
        out.append((ordinal, int(major.value), int(minor.value)))
    return out


def has_sm70_device() -> bool:
    """True if at least one visible CUDA device reports compute capability 7.0."""
    return any(major == 7 and minor == 0 for _, major, minor in _cuda_compute_capabilities())


def has_cuda_device() -> bool:
    return bool(_cuda_compute_capabilities())


def _discover_installed() -> dict[str, str]:
    """Map variant name → import path for native extensions sitting in this package."""
    pkg_dir = Path(__file__).resolve().parent
    found: dict[str, str] = {}
    for variant, short in (("generic", _GENERIC_MOD), ("cuda-sm70", _CUDA_SM70_MOD)):
        so_present = any(pkg_dir.glob(f"{short}*.so")) or any(pkg_dir.glob(f"{short}*.pyd"))
        spec = importlib.util.find_spec(f"fuzzybits.{short}")
        if so_present or spec is not None:
            found[variant] = f"fuzzybits.{short}"
    return found


def _select_variant(requested: str, installed: dict[str, str]) -> str:
    if not installed:
        raise ImportError(
            "no fuzzybits native extension found (expected fuzzybits._native_generic "
            "and/or fuzzybits._native_cuda_sm70). Reinstall the wheel."
        )
    if requested == "generic":
        if "generic" not in installed:
            raise ImportError("fuzzybits generic variant is not installed")
        return "generic"
    if requested in {"cuda-sm70", "cuda_sm70"}:
        if "cuda-sm70" not in installed:
            raise ImportError("fuzzybits cuda-sm70 variant is not installed")
        return "cuda-sm70"
    # auto: cuda-sm70 if installed *and* an sm_70 device is present, else generic.
    if "cuda-sm70" in installed and has_sm70_device():
        return "cuda-sm70"
    if "generic" in installed:
        return "generic"
    return next(iter(installed))


def _select_backend(requested: str) -> str:
    if requested == "cuda":
        return "cuda"
    if requested == "omp":
        return "omp"
    if has_cuda_device():
        return "cuda"
    return "omp"


def _apply_visibility(backend: str, device: str | int | None) -> None:
    # AdaptiveCpp always loads the OpenMP CPU backend; the mask still filters
    # CUDA/HIP/OpenCL/Level Zero.
    os.environ["ACPP_VISIBILITY_MASK"] = backend
    if device is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


def init(
    backend: Backend | None = None,
    device: str | int | None = None,
    variant: Variant | None = None,
) -> ModuleType:
    """Initialise AdaptiveCpp visibility and import the native extension.

    Must run before any other API that touches the SYCL runtime. Subsequent
    calls with the same arguments are no-ops; conflicting re-init raises.

    Parameters
    ----------
    backend:
        ``"cuda"``, ``"omp"``, or ``"auto"`` (CUDA if an NVIDIA device is
        visible, otherwise OpenMP). Overrides ``FUZZYBITS_BACKEND`` /
        ``MCSCRAM_BACKEND``.
    device:
        CUDA device ordinal or UUID forwarded to ``CUDA_VISIBLE_DEVICES``.
        ``FUZZYBITS_DEVICE`` is consulted when this is ``None``.
    variant:
        Compile-time wheel variant: ``"generic"``, ``"cuda-sm70"``, or
        ``"auto"``. Auto picks cuda-sm70 only when that extension is installed
        *and* an sm_70 GPU is present; otherwise generic. Overrides
        ``FUZZYBITS_VARIANT`` / ``MCSCRAM_VARIANT``.
    """
    global _NATIVE, _INITIALIZED, _BACKEND, _VARIANT, _DEVICE

    env_backend = (os.environ.get("FUZZYBITS_BACKEND") or os.environ.get("MCSCRAM_BACKEND") or "auto").lower()
    env_variant = (os.environ.get("FUZZYBITS_VARIANT") or os.environ.get("MCSCRAM_VARIANT") or "auto").lower()
    env_device = os.environ.get("FUZZYBITS_DEVICE") or os.environ.get("MCSCRAM_DEVICE")

    chosen_backend = _select_backend((backend or env_backend).lower())
    chosen_variant_req = (variant or env_variant).lower()
    chosen_device = device if device is not None else env_device

    if _INITIALIZED:
        if chosen_backend != _BACKEND or str(chosen_device) != str(_DEVICE):
            raise RuntimeError(
                "fuzzybits.init() already ran; AdaptiveCpp cannot switch "
                f"backend/device after import (active backend={_BACKEND!r}, "
                f"device={_DEVICE!r}). Start a new interpreter."
            )
        assert _NATIVE is not None
        return _NATIVE

    _ensure_windows_dll_search()
    _ensure_jit_cache()
    _ensure_runtime_defaults()
    _apply_visibility(chosen_backend, chosen_device)

    installed = _discover_installed()
    chosen_variant = _select_variant(chosen_variant_req, installed)
    native_mod = importlib.import_module(installed[chosen_variant])

    _NATIVE = native_mod
    _INITIALIZED = True
    _BACKEND = chosen_backend
    _VARIANT = chosen_variant
    _DEVICE = None if chosen_device is None else str(chosen_device)
    return native_mod


def native() -> ModuleType:
    """Return the loaded native module, initialising with defaults if needed."""
    if _NATIVE is None:
        return init()
    return _NATIVE


def active_backend() -> str | None:
    return _BACKEND


def active_variant() -> str | None:
    return _VARIANT


def active_device() -> str | None:
    return _DEVICE


def is_initialized() -> bool:
    return _INITIALIZED
