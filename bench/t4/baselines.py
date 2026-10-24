"""The kernels a T4 solution is judged against, each fn(x, w, offsets, out)
-> out (or None, having written `out`), computing the grouped GEMM of
reference.py with no host-side knowledge of the routing beyond what it
synchronises for (and pays for) inside the call.

    torch_loop        torch.mm per expert: hipBLASLt, the group bounds read to
                      the host once a call (reference.torch_impl)
    torch_grouped_mm  torch._grouped_mm (offs = the running row counts); on
                      ROCm torch's own grouped kernel or its per-group fallback
    triton            a grouped Triton GEMM tuned per shape (triton_grouped.py)
    aiter_gmm         aiter's Triton grouped GEMM (aiter.ops.triton.gmm) with
                      its gfx950 configs                                gfx950
    aiter_ck          aiter's CK MoE GEMM as aiter's fused_moe runs its second
                      stage (ck_moe_stage2_fwd, top-1, unit weights, the
                      block size and kernel get_2stage_cfgs picks), after
                      aiter's moe_sorting; weights pre-shuffled once
                      (shuffle_weight, a model-load step)               gfx950
    vllm_fused_moe    vLLM's fused-MoE Triton GEMM as vLLM runs it: experts per
                      row by searchsorted, vLLM's moe_align_block_size, its
                      config for the batch, invoke_fused_moe_triton_kernel
                                                     where vLLM is installed
    hk_ops_moe        hk's shipped MoE GEMM (hk.ops.moe.moe_gemm, chapter 21)
                      on sorted ids made on the GPU (vLLM's align where
                      installed, torch ops otherwise), 16-row blocks up to 32
                      rows an expert on average, 64-row above

`available(arch)` says which import here and why the others do not. score.py
checks every arm against fp32 before it is timed; a wrong or refusing arm is
dropped from "best baseline" for that shape, with the reason kept.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Tuple

ALL = ("torch_loop", "torch_grouped_mm", "triton", "aiter_gmm", "aiter_ck", "vllm_fused_moe",
       "hk_ops_moe")

#: name -> prepare(w): a baseline's untimed model-load step (score.py runs it
#: once per weight copy, as it runs a solution's).
PREPARE: Dict[str, Callable] = {}

_CACHE: Dict = {}


def _arange(n: int, device):
    import torch  # noqa: PLC0415

    key = ("arange", n, str(device))
    if key not in _CACHE:
        _CACHE[key] = torch.arange(n, device=device, dtype=torch.int32)
    return _CACHE[key]


def expert_of_row(offsets, rows: int):
    """(rows,) int32: the expert each row belongs to (searchsorted, no sync)."""
    import torch  # noqa: PLC0415

    return torch.searchsorted(offsets[1:], _arange(rows, offsets.device), right=True,
                              out_int32=True)


def sorted_from_offsets(offsets, block_m: int, rows: int):
    """vLLM's moe_align_block_size output for contiguous groups, made with
    torch ops on the GPU: (sorted_ids, expert_ids, num_tokens_post_padded),
    each expert's rows padded to a multiple of block_m with the sentinel
    `rows`, expert_ids -1 past the padded end."""
    import torch  # noqa: PLC0415

    e_n = offsets.numel() - 1
    sizes = offsets[1:] - offsets[:-1]
    padded = (sizes + (block_m - 1)) // block_m * block_m
    pend = torch.cumsum(padded, 0, dtype=torch.int32)
    pstart = pend - padded
    cap = -(-(rows + e_n * (block_m - 1)) // block_m) * block_m
    pos = _arange(cap, offsets.device)
    e = torch.searchsorted(pend, pos, right=True, out_int32=True)
    ec = e.clamp(max=e_n - 1)
    local = pos - pstart[ec]
    valid = (e < e_n) & (local < sizes[ec])
    sorted_ids = torch.where(valid, offsets[:-1][ec] + local, rows).to(torch.int32)
    eb = e[::block_m]
    expert_ids = torch.where(eb < e_n, eb, -1).to(torch.int32)
    return sorted_ids, expert_ids, pend[-1:].clone()


def _torch_loop(x, w, offsets, out):
    from . import reference  # noqa: PLC0415

    return reference.torch_impl(x, w, offsets, out)


def _torch_grouped_mm(x, w, offsets, out):
    import torch  # noqa: PLC0415

    return torch._grouped_mm(x, w.transpose(-2, -1), offs=offsets[1:], out_dtype=torch.bfloat16)


def _triton(x, w, offsets, out):
    from . import triton_grouped  # noqa: PLC0415

    return triton_grouped.grouped_mm(x, w, offsets, out)


def _aiter_gmm():
    from aiter.ops.triton.gmm import gmm  # noqa: PLC0415

    def run(x, w, offsets, out):
        sizes = offsets[1:] - offsets[:-1]
        return gmm(x, w.transpose(1, 2), sizes, existing_out=out)
    return run


def _aiter_ck():
    import torch  # noqa: PLC0415
    import aiter  # noqa: PLC0415
    from aiter import ActivationType, QuantType  # noqa: PLC0415
    from aiter.fused_moe import get_2stage_cfgs, get_padded_M, moe_sorting  # noqa: PLC0415
    from aiter.ops.shuffle import shuffle_weight  # noqa: PLC0415

    def prepare(w):
        return shuffle_weight(w, layout=(16, 16))

    def run(x, w, offsets, out):
        rows, k = x.shape
        e_n, n, _ = w.shape
        key = ("aiter_ck", rows, n, k, e_n)
        if key not in _CACHE:
            md = get_2stage_cfgs(get_padded_M(rows), n, k, e_n, 1, torch.bfloat16, torch.bfloat16,
                                 torch.bfloat16, QuantType.No, True, ActivationType.Silu, False,
                                 0, 0, True)
            w1 = torch.empty(1, device=x.device, dtype=torch.bfloat16).expand(e_n, 2 * k, n)
            _CACHE[key] = (md, w1, torch.ones(rows, 1, device=x.device, dtype=torch.float32))
        md, w1, ones = _CACHE[key]
        ids = expert_of_row(offsets, rows).view(rows, 1)
        sid, sw, seid, nvalid, buf = moe_sorting(ids, ones, e_n, n, torch.bfloat16, md.block_m)
        md.stage2(x.view(rows, 1, k), w1, w, sid, seid, nvalid, buf, 1, w2_scale=None,
                  a2_scale=None, block_m=md.block_m, sorted_weights=sw)
        return buf
    PREPARE["aiter_ck"] = prepare
    del aiter
    return run


def _vllm_fused_moe():
    import triton.language as tl  # noqa: PLC0415
    from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: PLC0415
        invoke_fused_moe_triton_kernel, try_get_optimal_moe_config)
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: PLC0415
        moe_align_block_size)

    def run(x, w, offsets, out):
        rows, k = x.shape
        e_n, n, _ = w.shape
        key = ("vllm_cfg", rows, n, k, e_n)
        if key not in _CACHE:
            _CACHE[key] = try_get_optimal_moe_config(tuple(w.shape), (e_n, k, n), 1, None, rows)
        cfg = _CACHE[key]
        ids = expert_of_row(offsets, rows).view(rows, 1)
        sorted_ids, expert_ids, ntp = moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], e_n)
        invoke_fused_moe_triton_kernel(
            x, w, out.view(rows, 1, n), None, None, None, sorted_ids, expert_ids, ntp,
            False, 1, cfg, tl.bfloat16, False, False, False, False, False, None, None)
        return out
    return run


def _hk_ops_moe(have_vllm_align: bool):
    from hk.ops import moe  # noqa: PLC0415

    align = None
    if have_vllm_align:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: PLC0415
            moe_align_block_size)
        align = moe_align_block_size

    def run(x, w, offsets, out):
        rows = x.shape[0]
        e_n = w.shape[0]
        bm = 16 if rows <= 32 * e_n else 64
        if not moe.takes(x, w, bm):
            raise ValueError(f"hk.ops.moe does not take N={w.shape[1]} K={w.shape[2]} at "
                             f"block_m {bm}")
        if align is not None:
            s, eids, ntp = align(expert_of_row(offsets, rows).view(rows, 1), bm, e_n)
        else:
            s, eids, ntp = sorted_from_offsets(offsets, bm, rows)
        moe.moe_gemm(x, w, out, s, eids, ntp, top_k=1, block_m=bm)
        return out
    return run


def available(arch: str, want=ALL) -> Tuple[Dict[str, Callable], Dict[str, str]]:
    """({name: fn}, {name: why it is not here})."""
    fns: Dict[str, Callable] = {}
    why: Dict[str, str] = {}
    have_vllm = False
    try:
        import vllm.model_executor.layers.fused_moe.moe_align_block_size  # noqa: PLC0415,F401

        have_vllm = True
    except Exception:  # noqa: BLE001
        pass
    for name in want:
        try:
            if name == "torch_loop":
                fns[name] = _torch_loop
            elif name == "torch_grouped_mm":
                import torch  # noqa: PLC0415

                if not hasattr(torch, "_grouped_mm"):
                    raise RuntimeError(f"torch {torch.__version__} has no _grouped_mm")
                fns[name] = _torch_grouped_mm
            elif name == "triton":
                from . import triton_grouped  # noqa: PLC0415,F401

                fns[name] = _triton
            elif name == "aiter_gmm":
                if arch != "gfx950":
                    raise RuntimeError("aiter is CDNA's")
                fns[name] = _aiter_gmm()
            elif name == "aiter_ck":
                if arch != "gfx950":
                    raise RuntimeError("aiter is CDNA's")
                fns[name] = _aiter_ck()
            elif name == "vllm_fused_moe":
                if not have_vllm:
                    raise RuntimeError("vLLM is not installed here")
                fns[name] = _vllm_fused_moe()
            elif name == "hk_ops_moe":
                fns[name] = _hk_ops_moe(have_vllm)
            else:
                raise ValueError(f"no baseline {name}")
        except Exception as e:  # noqa: BLE001 -- a missing library is a fact to record
            why[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
    return fns, why


def versions() -> Dict[str, str]:
    import torch  # noqa: PLC0415

    out = {"torch": torch.__version__, "hip": str(torch.version.hip)}
    for mod in ("triton", "aiter", "vllm", "flydsl"):
        try:
            m = __import__(mod)
            if getattr(m, "__version__", None):
                out[mod] = m.__version__
        except Exception:  # noqa: BLE001
            pass
    try:
        import importlib.metadata as md  # noqa: PLC0415

        for dist in ("amd-aiter", "hk-dsl"):
            try:
                out[dist] = md.version(dist)
            except md.PackageNotFoundError:
                pass
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("HK_T4_IMAGE"):
        out["image"] = os.environ["HK_T4_IMAGE"]
    return out
