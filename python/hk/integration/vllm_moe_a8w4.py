"""vLLM's MXFP4 MoE layers (AITER a8w4) on hk's MoE kernels at prefill sizes (`HK_VLLM_MOE_A8W4=1`, gfx950).

DeepSeek-V4.1-Flash's routed experts run through `aiter.fused_moe.fused_moe` on ROCm: FP8
activations against MXFP4 weights in AITER's shuffled layout, gate|up interleaved, SiLU with a
swiglu limit. vLLM imports that function inside its op at every call, so the patch wraps the
attribute itself. A call hk takes keeps AITER's own sort and fused quant-sort (`moe_sorting`,
`fused_dynamic_mxfp8_quant_moe_sort`). Then:
- hk's stage 1 (`moe_a8w4.stage1_sorted`) gathers each block's rows itself;
- the intermediate is quantized again by AITER's fused quant;
- hk's stage 2 (`stage2_sorted`) stores each weighted row by its (choice, token);
- the top-k sum is a plain sum over choices, deterministic, unlike AITER's atomics.
Everything else goes to AITER unchanged: decode sizes (M < `HK_MOE_A8W4_MIN_M`, default 2048),
CUDA-graph capture, other activations, quant types or gate modes, expert masks, biases, shared
experts.

`ENABLED` turns the route off and on at run time inside one process (an in-process A/B: the
compiled graph calls the same op either way).

    HK_PATCH_VLLM=1 HK_VLLM_MOE_A8W4=1 vllm serve <DeepSeek-V4.1-Flash>
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

from . import engine

_ORIG: Optional[Callable] = None
#: Route hk-able calls to hk (True) or leave every call to AITER (False).
ENABLED = True
MIN_M = int(os.environ.get("HK_MOE_A8W4_MIN_M", "2048"))
#: Token tiles a block (moe_sorting's block_size / 16): K1 c18-c23 measured 4 best.
MT = int(os.environ.get("HK_MOE_A8W4_MT", "4"))

PATCH = engine.Patch("hk_vllm::moe_a8w4", paths=("hk", "fallback"))   # capture: always AITER
STATS = PATCH.stats
_IDENT: Dict = {}


def _takes(hidden, w1, kw) -> bool:
    import torch  # noqa: PLC0415
    from aiter import ActivationType, QuantType  # noqa: PLC0415

    if not ENABLED or hidden.shape[0] < MIN_M or torch.cuda.is_current_stream_capturing():
        return False
    if getattr(torch, "float4_e2m1fn_x2", None) is None or w1.dtype != torch.float4_e2m1fn_x2:
        return False
    act, qt = kw.get("activation"), kw.get("quant_type")
    if act != ActivationType.Silu or qt != QuantType.per_1x32 or kw.get("doweight_stage1"):
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


def moe(hidden, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale, swiglu_limit: float):
    """The routed experts' output [M, hidden] bf16: AITER's sort and quant, hk's two stages."""
    import torch  # noqa: PLC0415
    from aiter.fused_moe import moe_sorting  # noqa: PLC0415
    from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort  # noqa: PLC0415

    from ..ops import moe_a8w4 as M  # noqa: PLC0415

    m, topk = topk_ids.shape
    e = w1.shape[0]
    inter = w1.shape[1] // 2
    bm = 16 * MT
    sid, sw, se, nv, _ = moe_sorting(topk_ids, topk_weight, e, hidden.shape[1], torch.bfloat16,
                                     block_size=bm)
    xq, xs = fused_dynamic_mxfp8_quant_moe_sort(hidden, sid, nv, m, topk, bm)
    w13 = w1.view(torch.uint8).view(-1, w1.shape[-1])
    w2b = w2.view(torch.uint8).view(-1, w2.shape[-1])
    h = M.stage1_sorted(xq, xs.view(torch.uint8), w13, w1_scale.view(torch.uint8), sid, se, nv,
                        inter=inter, mt=MT, pp=3, limit=float(swiglu_limit or 0.0) or 1e30)
    p = h.shape[0]
    key = (p, bm, h.device)
    if key not in _IDENT:
        _IDENT[key] = (torch.arange(p, dtype=torch.int32, device=h.device),
                       torch.tensor([p, p], dtype=torch.int32, device=h.device))
    ii, nn = _IDENT[key]
    hq, hs = fused_dynamic_mxfp8_quant_moe_sort(h, ii, nn, p, 1, bm)
    y2 = M.stage2_sorted(hq, hs.view(torch.uint8), w2b, w2_scale.view(torch.uint8), sid, se, nv,
                         sw, m, topk, e, mt=MT, pp=4)
    return y2.view(topk, m, -1).sum(0)


def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, *args, **kwargs):
    """Stands in for aiter.fused_moe.fused_moe."""
    names = ("expert_mask", "activation", "quant_type", "doweight_stage1", "w1_scale", "w2_scale",
             "a1_scale", "a2_scale")
    kw = dict(zip(names, args))
    kw.update(kwargs)
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
