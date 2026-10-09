"""vLLM on the generated attention kernel.

    import vllm                      # let vLLM import its modules first
    import hk.integration.vllm as hkv
    print(hkv.apply())               # before the engine is built

NOT VERIFIED INSIDE A VLLM SERVER. There is no vLLM build for RDNA3 on the
machine this was developed on, so this file has never run inside one. What *is*
verified is everything under it: `tests/hk/gpu/test_sdpa.py` checks the drop-in
against `F.scaled_dot_product_attention` elementwise on MHA, GQA, causal, a
non-dividing N and B=2, and checks that each unsupported call falls back to
torch rather than failing or quietly computing something else. What is
unverified is this file's reading of vLLM -- which modules hold their own
reference to SDPA, and whether the attention path in your build goes through
SDPA at all.

Read that last clause before expecting a speedup. vLLM on ROCm usually selects
a Triton or a CK/flash backend, neither of which calls
`F.scaled_dot_product_attention`; this patch then reports that it rebound
nothing, which is the truth and not an error. The backend it does reach is the
torch-SDPA one, plus any model code (vision towers, encoders, multimodal
adapters) that calls SDPA directly -- which on a VLM is most of the pixels.

`VLLM_ATTENTION_BACKEND=TORCH_SDPA` is how you make the decode path go through
it deliberately.
"""

from __future__ import annotations

from . import _common

PREFIXES = ("vllm",)

_APPLIED = None


def available() -> bool:
    return _common.importable("vllm")


def apply() -> str:
    """Patch vLLM's attention. Returns what happened, in one line."""
    global _APPLIED
    if not available():
        return "vllm is not importable; nothing patched"
    if _APPLIED is not None:
        return f"already applied: {_APPLIED}"
    report, names = _common.patch_sdpa(PREFIXES)
    _APPLIED = report
    if not names:
        report += ("; no vllm module held its own reference -- either vllm's "
                   "modules are not imported yet, or this build's attention "
                   "backend does not call SDPA")
    return report


def revert() -> str:
    global _APPLIED
    out = _common.unpatch_sdpa(PREFIXES)
    _APPLIED = None
    return out


def status() -> str:
    from ..ops import sdpa  # noqa: PLC0415

    if not available():
        return "vllm not importable"
    return "patched" if sdpa.patched() else "not patched"
