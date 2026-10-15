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

import importlib

from . import _common

PREFIXES = ("sglang",)

_APPLIED = None
_LINEAR = None

#: Where SGLang has kept UnquantizedLinearMethod, newest first.
_LINEAR_HOMES = ("sglang.srt.layers.quantization.unquant", "sglang.srt.layers.linear")


def apply_linear() -> str:
    """Route SGLang's unquantized linear layers through hk's GEMM op,
    `torch.ops.hk_vllm.linear` -- the op `vllm_gemm` defines, routed by the
    same measured table and falling back to F.linear, since without vLLM
    there is no other engine GEMM to fall back to. bf16 on the GPU only;
    every other call keeps SGLang's own apply. Returns what happened."""
    global _LINEAR
    if not available():
        return "sglang is not importable; linear not patched"
    if _LINEAR is not None:
        return "linear: already patched"
    from . import vllm_gemm  # noqa: PLC0415

    if vllm_gemm.routes() is None:
        return "linear: not patched -- no routing table for this arch"
    cls = None
    for home in _LINEAR_HOMES:
        try:
            cls = getattr(importlib.import_module(home), "UnquantizedLinearMethod", None)
        except ImportError:
            continue
        if cls is not None:
            break
    if cls is None:
        return f"linear: not patched -- no UnquantizedLinearMethod in {_LINEAR_HOMES}"
    vllm_gemm.register_op()
    orig = cls.apply

    def apply(self, layer, x, bias=None, *args, **kwargs):
        import torch  # noqa: PLC0415

        if x.is_cuda and x.dtype == torch.bfloat16 and not args and not kwargs:
            return torch.ops.hk_vllm.linear(x, layer.weight, bias)
        return orig(self, layer, x, bias, *args, **kwargs)

    cls.apply = apply
    _LINEAR = (cls, orig)
    return f"linear -> hk ({cls.__module__}.UnquantizedLinearMethod.apply)"


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
    global _APPLIED, _LINEAR
    out = _common.unpatch_sdpa(PREFIXES)
    _APPLIED = None
    if _LINEAR is not None:
        cls, orig = _LINEAR
        cls.apply = orig
        _LINEAR = None
        out += "; linear -> sglang"
    return out


def status() -> str:
    from ..ops import sdpa  # noqa: PLC0415

    if not available():
        return "sglang not importable"
    return "patched" if sdpa.patched() else "not patched"
