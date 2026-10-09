"""SGLang on the generated attention kernel.

    import sglang                      # let SGLang import its modules first
    import hk.integration.sglang as hks
    print(hks.apply())                 # before the engine is built

NOT VERIFIED INSIDE AN SGLANG SERVER, for the same reason as the vLLM module
next to it: there is no SGLang build for RDNA3 on the machine this was
developed on. `tests/hk/gpu/test_sdpa.py` verifies the drop-in itself against
`F.scaled_dot_product_attention`; this file's reading of SGLang's internals is
what is untested.

SGLang's RDNA3 path normally selects its Triton attention backend, which does
not call `F.scaled_dot_product_attention` -- so on a text model this patch will
honestly report that it rebound nothing. Where it does bite is
`--attention-backend torch_native` and the vision encoders in the multimodal
models, which call SDPA directly.
"""

from __future__ import annotations

from . import _common

PREFIXES = ("sglang",)

_APPLIED = None


def available() -> bool:
    return _common.importable("sglang")


def apply() -> str:
    """Patch SGLang's attention. Returns what happened, in one line."""
    global _APPLIED
    if not available():
        return "sglang is not importable; nothing patched"
    if _APPLIED is not None:
        return f"already applied: {_APPLIED}"
    report, names = _common.patch_sdpa(PREFIXES)
    _APPLIED = report
    if not names:
        report += ("; no sglang module held its own reference -- either "
                   "sglang's modules are not imported yet, or this build's "
                   "attention backend does not call SDPA")
    return report


def revert() -> str:
    global _APPLIED
    out = _common.unpatch_sdpa(PREFIXES)
    _APPLIED = None
    return out


def status() -> str:
    from ..ops import sdpa  # noqa: PLC0415

    if not available():
        return "sglang not importable"
    return "patched" if sdpa.patched() else "not patched"
