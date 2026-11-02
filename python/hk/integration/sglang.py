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

**Gemma-style RMSNorm** (`apply_norm`): SGLang's `GemmaRMSNorm` on ROCm
falls back to its native torch path -- eleven kernels a norm -- when neither
aiter nor vLLM is installed. There it goes through hk's weighted norm kernels
(`hk.ops.fused_add_rms_norm` / `rms_norm`), given the stored weight and
`weight_offset=1.0`, so the one is added in fp32 as SGLang's own kernels do.
Where SGLang has a fused path of its own (aiter, vLLM) it is left alone. The
plugin applies it before the model is built: SGLang resolves each norm's
forward on its first call.

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
_NORM = None

#: SGLang's GemmaRMSNorm route: calls hk took, and the calls it left to SGLang, by reason.
NORM_STATS: "Counter[str]" = None  # type: ignore[assignment]

#: Where SGLang keeps GemmaRMSNorm.
_NORM_HOME = "sglang.srt.layers.layernorm"

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
            return vllm_gemm.PATCH(x, layer.weight, bias)     # the op, or in eager code its body
        return orig(self, layer, x, bias, *args, **kwargs)

    cls.apply = apply
    _LINEAR = (cls, orig)
    return (f"linear -> hk ({cls.__module__}.UnquantizedLinearMethod.apply; "
            f"routes measured against {table.meta.get('engine', '?')})"
            + vllm_gemm.optional_status(table))


def _norm_refusal(self, x, residual) -> str:
    """Why hk leaves this call to SGLang, or "" if it takes it."""
    import torch  # noqa: PLC0415

    w = self.weight
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        return f"x {x.dtype} on {x.device.type}"
    if not x.is_contiguous():
        return "x not contiguous"
    if w.dtype != x.dtype:
        return f"weight {w.dtype}, x {x.dtype}"
    if w.dim() != 1 or w.shape[0] != x.shape[-1]:
        return f"weight {tuple(w.shape)} for rows of {x.shape[-1]}"
    if residual is not None and (residual.shape != x.shape or residual.dtype != x.dtype
                                 or not residual.is_contiguous()):
        return "residual not x's contiguous shape and dtype"
    return ""


_NORM_OPS: dict = {}


def _norm_op(kernel, consts):
    """`kernel` at `consts` as a torch op (hk.runtime.torch_ext.launcher), memoized: it launches
    on torch's current stream. hk.ops' own calls go through the pybind entry point, which on
    RDNA launches on the null stream -- unordered against the stream SGLang runs its model on,
    so the norm would read its input before the kernel that writes it has finished."""
    key = (kernel.name, tuple(sorted(consts.items())))
    op = _NORM_OPS.get(key)
    if op is None:
        from ..runtime.torch_ext import launcher  # noqa: PLC0415

        tag = "_".join(f"{k}{v}" for k, v in key[1]).replace(".", "p").replace("-", "m")
        op = _NORM_OPS[key] = launcher(kernel, ns="hk_sglang", name=f"{kernel.name}_{tag}",
                                       **consts)
    return op


def gemma_rms_norm(x, residual, weight, eps: float):
    """y = x_hat * (1 + weight), with the residual add fused when `residual` is given: hk's
    weighted norm kernels at weight offset 1, out of place -- (y, x + residual) or y -- and on
    torch's current stream. The plan is hk.ops.norm's (strip, else tile)."""
    import torch  # noqa: PLC0415

    from ..ops import norm  # noqa: PLC0415

    rows, cols = norm._rows_cols(x)
    kind = "rmsw" if residual is None else "addrmsw"
    out = torch.empty_like(x)
    res_out = None if residual is None else torch.empty_like(x)
    got = norm._weighted_strip(kind, x, rows, cols, 1.0)
    if got is not None:
        k, chunks, run, exact, reserve = got
        consts = dict(CHUNKS=chunks, RUN=run, EXACT=exact, RESERVE=reserve, EPS=eps)
        view = (lambda t: norm._flat(t, rows, cols))
        w = weight.view(1, cols)
    else:
        k, tpw, fold, exact = norm._resolve_weighted(kind, x.dtype, cols, rows, 1.0)
        consts = dict(TPW=tpw, FOLD=fold, EXACT=exact, EPS=eps)
        view = (lambda t: t.view(rows * fold, cols // fold))
        w = weight.view(fold, cols // fold)
    op = _norm_op(k, consts)
    if residual is None:
        op(view(x), w, view(out))
        return out
    op(view(x), view(residual), w, view(out), view(res_out))
    return out, res_out


def apply_norm() -> str:
    """Route SGLang's GemmaRMSNorm through hk's weighted norm kernels where
    SGLang would take its native torch path (no aiter, no vLLM): the stored
    weight at weight offset 1, the residual add fused, out of place as
    SGLang's HIP paths are, on torch's current stream (gemma_rms_norm). Every
    other call keeps SGLang's own forward, counted in NORM_STATS with its
    reason. Returns what happened."""
    global _NORM, NORM_STATS
    from collections import Counter  # noqa: PLC0415

    if not available():
        return "sglang is not importable; norm not patched"
    if _NORM is not None:
        return "norm: already patched"
    try:
        L = importlib.import_module(_NORM_HOME)
    except ImportError as e:
        return f"norm: not patched -- {_NORM_HOME}: {e}"
    cls = getattr(L, "GemmaRMSNorm", None)
    if cls is None:
        return f"norm: not patched -- no GemmaRMSNorm in {_NORM_HOME}"
    if getattr(L, "_use_aiter", False) or getattr(L, "_has_vllm_rms_norm", False):
        return "norm: not patched -- SGLang has a fused GemmaRMSNorm here (aiter / vLLM)"

    NORM_STATS = Counter()
    orig = cls.forward_hip

    def forward_hip(self, x, residual=None, post_residual_addition=None):
        why = _norm_refusal(self, x, residual)
        if why:
            NORM_STATS[why] += 1
            return orig(self, x, residual, post_residual_addition)
        NORM_STATS["hk"] += 1
        if residual is not None and post_residual_addition is not None:
            residual = residual + post_residual_addition
        return gemma_rms_norm(x, residual, self.weight, self.variance_epsilon)

    cls.forward_hip = forward_hip
    _NORM = (cls, orig)
    return f"norm -> hk ({_NORM_HOME}.GemmaRMSNorm.forward_hip, weight offset 1 in fp32)"


def norm_stats() -> str:
    return f"sglang_norm {dict(NORM_STATS or {})}"


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
    global _APPLIED, _LINEAR, _NORM
    out = _common.unpatch_sdpa(PREFIXES)
    _APPLIED = None
    if _LINEAR is not None:
        from . import vllm_gemm  # noqa: PLC0415

        cls, orig = _LINEAR
        cls.apply = orig
        _LINEAR = None
        vllm_gemm.set_routes(None)
        out += "; linear -> sglang"
    if _NORM is not None:
        cls, orig = _NORM
        cls.forward_hip = orig
        _NORM = None
        out += "; norm -> sglang"
    return out


def status() -> str:
    from ..ops import sdpa  # noqa: PLC0415

    if not available():
        return "sglang not importable"
    return "patched" if sdpa.patched() else "not patched"
