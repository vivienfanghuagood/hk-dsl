"""vLLM's MXFP4 MoE layers (AITER a8w4) on hk's MoE kernels at prefill sizes (`HK_VLLM_MOE_A8W4=1`, gfx950).

DeepSeek-V4.1-Flash's routed experts run through `aiter.fused_moe.fused_moe` on ROCm: FP8
activations against MXFP4 weights in AITER's shuffled layout, gate|up interleaved, SiLU with a
swiglu limit. vLLM imports that function inside its op at every call, so the patch wraps the
attribute itself. A call hk takes keeps AITER's own sort and fused quant-sort (`moe_sorting`,
`fused_dynamic_mxfp8_quant_moe_sort`). Then:
- hk's stage 1 (`moe_a8w4.stage1_sorted`) gathers each block's rows itself;
- its epilogue MX-quantizes the intermediate itself (`hk.mx_quant_store`, AITER's rule);
- hk's stage 2 (`stage2_sorted`, the block's h resident in LDS) stores each weighted row by its
  (choice, token);
- the top-k sum is a plain sum over choices, deterministic, unlike AITER's atomics.
Everything else goes to AITER unchanged: M < `HK_MOE_A8W4_MIN_M` (default 8192, see MIN_M),
CUDA-graph capture, other activations, quant types or gate modes, expert masks, biases, shared
experts.

`ENABLED` turns the route off and on at run time inside one process (an in-process A/B: the
compiled graph calls the same op either way).

    HK_PATCH_VLLM=1 HK_VLLM_MOE_A8W4=1 vllm serve <DeepSeek-V4.1-Flash>
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from ..runtime import hostpath as _hostpath
from . import engine

_ORIG: Optional[Callable] = None
#: Route hk-able calls to hk (True) or leave every call to AITER (False).
ENABLED = True
#: The smallest M hk takes: where it wins. Against AITER on DSV4.1's TP2 shapes (K1, c32-c34):
#: 1.18x at 8192, 1.15-1.16x at 16384, 1.06-1.08x at 5120, and 0.90x at 2048. At small M both
#: are bound by streaming every expert's weights (2.26 GB); AITER is within 10% of that floor,
#: and hk re-reads x across its column blocks. So the route starts at 8192.
MIN_M = int(os.environ.get("HK_MOE_A8W4_MIN_M", "8192"))
#: Token tiles a block (moe_sorting's block_size / 16): K1 c18-c23 measured 4 best.
MT = int(os.environ.get("HK_MOE_A8W4_MT", "4"))

PATCH = engine.Patch("hk_vllm::moe_a8w4", paths=("hk", "fallback"))   # capture: always AITER
STATS = PATCH.stats


#: aiter.fused_moe.fused_moe's parameters this route reads by name (_takes, fused_moe). A
#: nightly that renames or drops one would silently send every call to AITER -- or, as a bare
#: (*args, **kwargs) wrapper once did with gate_mode, change the stock path too.
#: tests/hk/gpu/test_engine_contract.py holds the installed AITER to it.
AITER_PARAMS = ("activation", "quant_type", "doweight_stage1", "gate_mode", "expert_mask",
                "a1_scale", "a2_scale", "num_local_tokens", "bias1", "bias2", "shared_w1",
                "shared_w2", "quant_dtype_a", "hidden_pad", "intermediate_pad", "w1_scale",
                "w2_scale", "swiglu_limit")


#: What `_takes` compares against, looked up once: (torch, SiLU, per-1x32 quant, the FP4 dtype or
#: None). An import statement in a function body is a module-dict lookup and an attribute walk
#: on every call, and vLLM makes this call on every MoE layer, routed or not.
_CONSTS: list = []


def _consts():
    if not _CONSTS:
        import torch  # noqa: PLC0415
        from aiter import ActivationType, QuantType  # noqa: PLC0415

        _CONSTS.extend((torch, ActivationType.Silu, QuantType.per_1x32,
                        getattr(torch, "float4_e2m1fn_x2", None)))
    return _CONSTS


def _consts_legacy():
    """`_consts` as the imports in `_takes`'s body did it before the host fast paths: every call
    (HK_HOST_FASTPATH=0)."""
    import torch  # noqa: PLC0415
    from aiter import ActivationType, QuantType  # noqa: PLC0415

    return torch, ActivationType.Silu, QuantType.per_1x32, getattr(torch, "float4_e2m1fn_x2", None)


def _takes(hidden, w1, kw) -> bool:
    torch, silu, per_1x32, fp4 = _consts() if _hostpath.ENABLED else _consts_legacy()
    if not ENABLED or hidden.shape[0] < MIN_M or torch.cuda.is_current_stream_capturing():
        return False
    if fp4 is None or w1.dtype != fp4:
        return False
    act, qt = kw.get("activation"), kw.get("quant_type")
    if act != silu or qt != per_1x32 or kw.get("doweight_stage1"):
        return False
    if kw.get("gate_mode", "") not in ("interleave",):
        return False
    for k in ("expert_mask", "a1_scale", "a2_scale", "num_local_tokens", "bias1", "bias2",
              "shared_w1", "shared_w2", "quant_dtype_a"):
        if kw.get(k) is not None:
            return False
    if kw.get("hidden_pad", 0) or kw.get("intermediate_pad", 0):
        return False
    inter = w1.shape[1] // 2
    # stage 1: 192-column workgroups (pp 3); stage 2: K = inter, whole 128-wide steps, at least four
    return (inter % 384 == 0 and inter >= 768 and hidden.shape[1] % 512 == 0
            and hidden.dtype == torch.bfloat16)


_FNS: list = []


def _fns():
    """(torch, moe_sorting, fused_dynamic_mxfp8_quant_moe_sort, hk.ops.moe_a8w4), imported once."""
    if not _FNS:
        import torch  # noqa: PLC0415
        from aiter.fused_moe import moe_sorting  # noqa: PLC0415
        from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort  # noqa: PLC0415

        from ..ops import moe_a8w4  # noqa: PLC0415

        _FNS.extend((torch, moe_sorting, fused_dynamic_mxfp8_quant_moe_sort, moe_a8w4))
    return _FNS


def _fns_legacy():
    """`_fns` as `moe`'s body imported them before the host fast paths: every call."""
    import torch  # noqa: PLC0415
    from aiter.fused_moe import moe_sorting  # noqa: PLC0415
    from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort  # noqa: PLC0415

    from ..ops import moe_a8w4  # noqa: PLC0415

    return torch, moe_sorting, fused_dynamic_mxfp8_quant_moe_sort, moe_a8w4


def moe(hidden, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale, swiglu_limit: float):
    """The routed experts' output [M, hidden] bf16: AITER's sort and quant, hk's two stages."""
    torch, moe_sorting, fused_dynamic_mxfp8_quant_moe_sort, M = (
        _fns() if _hostpath.ENABLED else _fns_legacy())

    m, topk = topk_ids.shape
    e = w1.shape[0]
    inter = w1.shape[1] // 2
    bm = 16 * MT
    sid, sw, se, nv, _ = moe_sorting(topk_ids, topk_weight, e, hidden.shape[1], torch.bfloat16,
                                     block_size=bm)
    xq, xs = fused_dynamic_mxfp8_quant_moe_sort(hidden, sid, nv, m, topk, bm)
    w13 = w1.view(torch.uint8).view(-1, w1.shape[-1])
    w2b = w2.view(torch.uint8).view(-1, w2.shape[-1])
    hq, hs = M.stage1_sorted(xq, xs.view(torch.uint8), w13, w1_scale.view(torch.uint8), sid, se, nv,
                             inter=inter, mt=MT, pp=3, limit=float(swiglu_limit or 0.0) or 1e30,
                             mxq=True)     # h MX-quantized by stage 1's epilogue (K1 c31)
    d = w2.shape[1]
    # h resident in LDS (K1 c32) while a block's h (16 * MT rows x inter bytes) leaves room for two
    # workgroups a CU: TP2's inter 1152 (1.05-1.07x faster stage 2), not TP1's 2304 (one a CU, slower)
    res = 5 if inter % 384 == 0 and d % (256 * 5) == 0 and 16 * MT * inter <= 80 * 1024 else 0
    y2 = M.stage2_sorted(hq, hs.view(torch.uint8), w2b, w2_scale.view(torch.uint8), sid, se, nv,
                         sw, m, topk, e, mt=MT, pp=4, resident=res)
    return y2.view(topk, m, -1).sum(0)


#: AITER's fused_moe's signature, read once (apply_patch): (the signature, its positional
#: parameters after the five named here, every parameter's name, whether it takes **kwargs).
#: inspect.signature walks the function on every call -- most of the 54 us this wrapper cost
#: on each MoE layer on the MI355X.
_SIG: list = []


def _sig():
    if not _SIG or _SIG[0][0] is not _ORIG:
        import inspect  # noqa: PLC0415

        sig = inspect.signature(_ORIG)
        ps = list(sig.parameters.values())
        pos = tuple(p.name for p in ps if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
        if any(p.kind == p.VAR_POSITIONAL for p in ps):
            pos = None                      # *args: leave every call to bind()
        _SIG[:] = [(_ORIG, sig, pos[5:] if pos else None, frozenset(sig.parameters),
                    any(p.kind == p.VAR_KEYWORD for p in ps))]
    return _SIG[0]


def _arguments(args, kwargs, five) -> dict:
    """inspect.signature(_ORIG).bind(*five, *args, **kwargs).arguments with **kwargs folded in --
    the call's arguments by name -- without bind() where the call is one bind would accept as
    is: no more positionals than there are parameters, no name given twice, no unknown name
    unless the function takes **kwargs. Anything else goes to bind(), which raises as before."""
    orig, sig, pos, names, var_kw = _sig()
    if pos is not None and len(args) <= len(pos) and kwargs.keys().isdisjoint(pos[:len(args)]) \
            and kwargs.keys().isdisjoint(_FIVE) and (var_kw or kwargs.keys() <= names):
        kw = dict(zip(_FIVE, five))
        kw.update(zip(pos, args))
        kw.update(kwargs)
        return kw
    kw = sig.bind(*five, *args, **kwargs).arguments
    kw.update(kw.pop("kwargs", {}))
    return kw


_FIVE = ("hidden_states", "w1", "w2", "topk_weight", "topk_ids")


def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, *args, **kwargs):
    """Stands in for aiter.fused_moe.fused_moe, with its signature (apply_patch): vLLM probes
    that signature for `gate_mode` / `quant_dtype_a` and silently drops them when absent -- AITER
    then reads interleaved weights as separated ones, in the stock path as well."""
    if _hostpath.ENABLED:
        kw = _arguments(args, kwargs, (hidden_states, w1, w2, topk_weight, topk_ids))
    else:
        import inspect  # noqa: PLC0415

        kw = inspect.signature(_ORIG).bind(hidden_states, w1, w2, topk_weight, topk_ids, *args,
                                           **kwargs).arguments
        kw.update(kw.pop("kwargs", {}))
    if _takes(hidden_states, w1, kw):
        PATCH.count("hk")
        return moe(hidden_states, w1, w2, topk_weight, topk_ids, kw["w1_scale"], kw["w2_scale"],
                   kw.get("swiglu_limit", 0.0))
    PATCH.count("fallback")
    return _ORIG(hidden_states, w1, w2, topk_weight, topk_ids, *args, **kwargs)


def apply_patch() -> str:
    """Wrap aiter.fused_moe.fused_moe. Returns what happened, in one line."""
    global _ORIG
    from . import _common  # noqa: PLC0415

    if not _common.importable("aiter"):
        return "moe_a8w4: aiter is not importable; not patched"
    if _ORIG is not None:
        return "moe_a8w4: already patched"
    from ..target import current_arch  # noqa: PLC0415

    if current_arch() != "gfx950":
        return f"moe_a8w4: not patched -- gfx950's scaled MFMA, not {current_arch()}"
    import aiter.fused_moe as afm  # noqa: PLC0415

    _ORIG = afm.fused_moe
    fused_moe.__wrapped__ = _ORIG        # inspect.signature(fused_moe) is AITER's
    afm.fused_moe = fused_moe
    names = _common.rebind(("vllm", "aiter"), _ORIG, fused_moe)
    return (f"moe_a8w4 -> hk for AITER a8w4 MoE calls of M >= {MIN_M} "
            f"({len(names)} other bindings rebound)")


def revert() -> str:
    global _ORIG
    if _ORIG is None:
        return "moe_a8w4: not patched"
    import aiter.fused_moe as afm  # noqa: PLC0415
    from . import _common  # noqa: PLC0415

    afm.fused_moe = _ORIG
    _common.rebind(("vllm", "aiter"), fused_moe, _ORIG)
    _ORIG = None
    return "moe_a8w4 -> aiter"


def status() -> str:
    return "not patched" if _ORIG is None else PATCH.summary()
