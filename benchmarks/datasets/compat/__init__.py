"""Dataset-side compatibility shims. The MEF loader stays strict."""

from .generic_pwr import (
    SHIM_ID,
    SHIM_VERSION,
    PatchResult,
    matching_jsinp,
    patch_generic_pwr,
)

__all__ = [
    "SHIM_ID",
    "SHIM_VERSION",
    "PatchResult",
    "matching_jsinp",
    "patch_generic_pwr",
]
