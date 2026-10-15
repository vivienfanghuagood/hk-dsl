"""SGLang on hk's linear layers, and on the generated attention kernel.

**Linear layers** (`apply_linear`), measured inside SGLang 0.5.18 with
AMD-AIM/sglang-radeon on a W7900D: every unquantized bf16 linear goes through
`torch.ops.hk_vllm.linear`, routed by a table measured against the F.linear
SGLang calls on ROCm. A server turns it on with `HK_SGLANG_LINEAR=1`, through
the `sglang.srt.plugins` entry point (`_sglang_plugin`), which SGLang loads in
the scheduler process where the model runs:

    HK_SGLANG_LINEAR=1 python -m sglang.launch_server --model-path Qwen/Qwen3-8B \\
        --attention-backend triton --disable-cuda-graph

In one process (an offline engine, a benchmark), call it after importing
SGLang and before the model is built:

    import sglang
    import hk.integration.sglang as hks
    print(hks.apply_linear())

**Attention** (`apply`) is still not verified inside SGLang.
`tests/hk/gpu/test_sdpa.py` verifies the drop-in itself against
`F.scaled_dot_product_attention`. SGLang's RDNA3 path selects its Triton
attention backend, which does not call `F.scaled_dot_product_attention`, so on
a text model this patch will honestly report that it rebound nothing. Where it
does bite is `--attention-backend torch_native` and the vision encoders in the
multimodal models, which call SDPA directly.
"""

from __future__ import annotations

import importlib

from . import _common

PREFIXES = ("sglang",)

#: Where a routing table override for SGLang's linear comes from.
ROUTES_ENV = "HK_SGLANG_ROUTES"

_APPLIED = None
_LINEAR = None

#: Where SGLang has kept UnquantizedLinearMethod, newest first.
_LINEAR_HOMES = ("sglang.srt.layers.quantization.unquant", "sglang.srt.layers.linear")


def routes():
    """The table SGLang's linear is routed by: one measured against F.linear,
    which SGLang's linear calls on ROCm (routes_sglang_linear_<arch>.json, or
    $HK_SGLANG_ROUTES), where there is one; else vLLM's. vLLM's fallback rows
    were measured against vLLM's skinny GEMMs, so under SGLang they keep
    F.linear at Ms where the generated GEMV beats it."""
    from ..target import current_arch  # noqa: PLC0415
    from . import engine, vllm_gemm  # noqa: PLC0415

    return (engine.shipped_table("sglang_linear", current_arch(), ROUTES_ENV)
            or vllm_gemm.routes())


def apply_linear() -> str:
    """Route SGLang's unquantized linear layers through hk's GEMM op,
    `torch.ops.hk_vllm.linear` -- the op `vllm_gemm` defines, routed by
    `routes()` and falling back to F.linear, since without vLLM there is no
    other engine GEMM to fall back to. bf16 on the GPU only; every other call
    keeps SGLang's own apply. Returns what happened."""
    global _LINEAR
    if not available():
        return "sglang is not importable; linear not patched"
    if _LINEAR is not None:
        return "linear: already patched"
    from . import vllm_gemm  # noqa: PLC0415

    table = routes()
    if table is None:
        return "linear: not patched -- no routing table for this arch"
    vllm_gemm.set_routes(table)
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
    return (f"linear -> hk ({cls.__module__}.UnquantizedLinearMethod.apply; "
            f"routes measured against {table.meta.get('engine', '?')})")


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
        from . import vllm_gemm  # noqa: PLC0415

        cls, orig = _LINEAR
        cls.apply = orig
        _LINEAR = None
        vllm_gemm.set_routes(None)
        out += "; linear -> sglang"
    return out


def status() -> str:
    from ..ops import sdpa  # noqa: PLC0415

    if not available():
        return "sglang not importable"
    return "patched" if sdpa.patched() else "not patched"
