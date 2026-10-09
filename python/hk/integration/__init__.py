"""Putting the generated kernels inside a serving framework.

    python3 -c "import hk.integration as hki; print(hki.apply())"

`apply()` detects what is importable and patches it; each backend reports what
it took and what it left alone, because a patch that silently did nothing is
worse than one that refused.

                        --- what gets patched, and why ---

**Attention, yes.** On Radeon every framework's attention path reaches
`F.scaled_dot_product_attention`, which on ROCm is aotriton's `attn_fwd`: flat
at 20-21 TFLOPs on a W7900D from N=4K to N=32K. The generated kernel is 61-65.
That is the patch worth making and it is the one these modules make.

**RMSNorm, no -- not yet.** Every framework's RMSNorm multiplies by a learned
weight, and `hk.ops.rmsnorm` is deliberately unweighted (the weight belongs to
whatever fuses next). Patching it would mean an extra elementwise pass over the
activations, which loses to the framework's own fused kernel. The missing piece
is a weighted RMSNorm kernel, not a patch.

**SiLU-mul, no -- not yet.** `SiluAndMul` is handed one `(..., 2d)` tensor and
slices it; both halves are then non-contiguous, and `hk.ops.silu_mul` has no
stride support, so the patch would pay two copies to save one pass. The missing
piece is column-offset globals, not a patch.

Those two are written down rather than attempted because the alternative -- a
patch that is slower than what it replaced -- is the specific failure that makes
people distrust a kernel library.
"""

from __future__ import annotations

from typing import Dict, List

from . import sglang, vllm

__all__ = ["apply", "revert", "status", "vllm", "sglang", "BACKENDS"]

BACKENDS = {"vllm": vllm, "sglang": sglang}


def apply(backends: List[str] = ()) -> Dict[str, str]:
    """Patch every framework that is importable. Returns name -> what happened.

    Never raises for a framework that is absent or has moved on: the report
    says so and the process keeps its own kernels.
    """
    names = list(backends) or list(BACKENDS)
    out = {}
    for n in names:
        mod = BACKENDS.get(n)
        if mod is None:
            out[n] = f"unknown backend; have {sorted(BACKENDS)}"
            continue
        try:
            out[n] = mod.apply()
        except Exception as e:  # noqa: BLE001 -- a patch must not take the server down
            out[n] = f"not patched: {type(e).__name__}: {e}"
    return out


def revert(backends: List[str] = ()) -> Dict[str, str]:
    names = list(backends) or list(BACKENDS)
    return {n: BACKENDS[n].revert() for n in names if n in BACKENDS}


def status() -> Dict[str, str]:
    return {n: m.status() for n, m in BACKENDS.items()}
