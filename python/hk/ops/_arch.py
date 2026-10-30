"""The GPUs an hk.ops kernel family builds for, and one refusal for the rest.

gfx942 (MI300X / MI325X) is compile-only (docs/supported-hardware.md). The
row-wise and elementwise ops build for it on their tile fallback, but the GEMM,
attention, paged attention, decode GEMM, MoE and W4A16 kernels are RDNA's, with
a gfx950 version beside them, and their RDNA kernels do not build for gfx942:
without this check a call got as far as hipcc and failed there, on a missing
include/cdna3 member. The check is at each entry point, not in the kernel
builders, so tools that probe what builds where (tools/cdna-probes) still can.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..target import current_arch

#: GEMM, attention, paged attention, decode GEMM, MoE and W4A16.
RDNA_AND_GFX950: Tuple[str, ...] = ("gfx1100", "gfx1201", "gfx950")


def why_not(op: str, arches: Tuple[str, ...] = RDNA_AND_GFX950,
            arch: Optional[str] = None) -> str:
    """Why `op` has no kernel for `arch` (the current one by default), or ''."""
    arch = arch or current_arch()
    if arch in arches:
        return ""
    note = (" gfx942 (MI300X / MI325X) is compile-only: docs/supported-hardware.md says what "
            "builds there." if arch == "gfx942" else "")
    return f"{op} has no kernel for {arch}; it runs on {', '.join(arches)}.{note}"


def require(op: str, arches: Tuple[str, ...] = RDNA_AND_GFX950,
            arch: Optional[str] = None) -> None:
    """Raise NotImplementedError (why_not's reason) unless `op` runs on `arch`."""
    why = why_not(op, arches, arch)
    if why:
        raise NotImplementedError(why)
