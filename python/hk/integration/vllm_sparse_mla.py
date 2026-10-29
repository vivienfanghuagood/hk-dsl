"""DeepSeek-V4.1's sparse MLA prefill on hk's kernel (`HK_VLLM_SPARSE_MLA=1`, gfx950).

vLLM's ROCm path (`vllm/models/deepseek_v41/amd/rocm.py`, `DeepseekV41ROCMAiterMLAAttention._forward_prefill`)
gathers each chunk's KV rows out of its fp8 caches into a bf16 workspace and calls
`rocm_sparse_attn_prefill` -- AITER's opus kernel at T >= 1024 queries, vLLM's Triton kernel below.
On gfx950 the layer's output is then inverse-RoPE'd and MXFP8-quantized for wo_a in one more pass
(`rocm_inverse_rope_mxfp8_rows`). The patch (hk.ops.sparse_mla):

* `rocm_sparse_attn_prefill`: a call of T >= MIN_T queries, 32 heads of 512, bf16, with a sink, runs
  hk's kernel on the same ragged rows vLLM builds; anything else runs vLLM's;
* `_forward_prefill` notes, for its chunk calls, where the layer's MXFP8 rows, positions and cos/sin
  are; a chunk hk takes then writes its MXFP8 rows itself, from O in registers (bit for bit vLLM's
  quant of hk's bf16 rows), and no bf16 rows -- vLLM's are a scratch buffer for that quant;
* `rocm_inverse_rope_mxfp8_rows` quantizes only the rows no hk call wrote (none, when every chunk
  went to hk).

Measured (DSV4.1-Flash TP2 real-weight captures, one MI355X): with the MXFP8
rows, top-k layers 1.39x and sliding-window layers 1.97x opus + vLLM's quant at T 8192 (1.22x and
1.12x AITER's fused fp8 kernel); attention alone 1.11-1.13x opus. `ENABLED` toggles the route at run time (an in-process A/B).

    HK_PATCH_VLLM=1 HK_VLLM_SPARSE_MLA=1 vllm serve <DeepSeek-V4.1-Flash>
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Dict, List, Optional, Tuple

from . import engine

#: The fewest queries a call needs: one query a workgroup, so a small call leaves the machine
#: half empty. Measured at T 512 and up hk beats opus (1.10x at 512 on a top-k layer); at T 128-256
#: (the decoder-replay layers) opus is faster (27-33 against 41-42 us), vLLM's Triton slower (70-77).
MIN_T = int(os.environ.get("HK_SPARSE_MLA_MIN_T", "512"))
#: Write the MXFP8 rows in the kernel (True) or leave them to vLLM's pass (False).
FUSE = os.environ.get("HK_SPARSE_MLA_FUSE", "1") not in ("0", "", "off", "no", "false")
#: Route hk-able calls to hk (True) or leave every call to vLLM (False).
ENABLED = True

PATCH = engine.Patch("hk_vllm::sparse_mla", paths=("hk", "hk_mxfp8", "fallback"))
STATS = PATCH.stats

_OPS_MOD = "vllm.v1.attention.ops.rocm_aiter_mla_sparse"
_ROCM_MOD = "vllm.models.deepseek_v41.amd.rocm"
_ORIG: Dict[str, Callable] = {}
_CTX = threading.local()
_CS32: Dict[int, object] = {}


class _Layer:
    """One `_forward_prefill` call's MXFP8 destination, for its chunk calls."""

    __slots__ = ("q", "positions", "cos_sin", "mxd", "mxs", "done")

    def __init__(self, q, positions, cos_sin, mxd, mxs):
        self.q, self.positions, self.cos_sin, self.mxd, self.mxs = q, positions, cos_sin, mxd, mxs
        self.done: List[Tuple[int, int]] = []


def _cos_sin_fp32(cs):
    """The rotary cos/sin cache in fp32 (vLLM's quant widens a bf16 cache on load: the same values)."""
    import torch  # noqa: PLC0415

    if cs.dtype == torch.float32 and cs.is_contiguous():
        return cs
    key = id(cs)
    got = _CS32.get(key)
    if got is None or got[0] is not cs:
        got = (cs, cs.float().contiguous())
        _CS32[key] = got
    return got[1]


def _takes(q, kv, attn_sink, output, head_dim, rope_head_dim) -> bool:
    import torch  # noqa: PLC0415

    from ..ops import sparse_mla as SM  # noqa: PLC0415

    return (ENABLED and q.dim() == 3 and q.shape[0] >= MIN_T and tuple(q.shape[1:]) == (SM.H, SM.D)
            and head_dim == SM.D and rope_head_dim == SM.D - SM.NOPE and q.dtype == torch.bfloat16
            and kv.dtype == torch.bfloat16 and kv.dim() == 3 and kv.shape[1] == 1 and kv.shape[2] == SM.D
            and output.dtype == torch.bfloat16 and tuple(output.shape) == tuple(q.shape)
            and q.is_contiguous() and output.is_contiguous() and kv.stride(2) == 1
            and attn_sink is not None and attn_sink.dtype == torch.float32 and attn_sink.numel() >= SM.H
            and q.is_cuda)


def rocm_sparse_attn_prefill(q, kv, indices, topk_length, scale, head_dim, nope_head_dim, rope_head_dim,
                             attn_sink, output, ragged_indices=None, ragged_indptr=None):
    """Stands in for vLLM's rocm_sparse_attn_prefill, with its signature."""
    orig = _ORIG["prefill"]
    if not _takes(q, kv, attn_sink, output, head_dim, rope_head_dim):
        PATCH.count("fallback")
        return orig(q, kv, indices, topk_length, scale, head_dim, nope_head_dim, rope_head_dim, attn_sink,
                    output, ragged_indices=ragged_indices, ragged_indptr=ragged_indptr)
    import importlib  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from ..ops import sparse_mla as SM  # noqa: PLC0415

    ops = importlib.import_module(_OPS_MOD)
    if ragged_indices is None or ragged_indptr is None:
        ind2 = indices.reshape(indices.shape[0], -1)
        ragged_indices, ragged_indptr = ops.build_ragged_indices_from_dense(
            ind2, topk_length if topk_length is not None else (ind2 >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=kv.shape[0])
    ind = ragged_indices if ragged_indices.dtype == torch.int32 and ragged_indices.is_contiguous() \
        else ragged_indices.to(torch.int32).contiguous()
    ptr = ragged_indptr if ragged_indptr.dtype == torch.int32 and ragged_indptr.is_contiguous() \
        else ragged_indptr.to(torch.int32).contiguous()
    sink = attn_sink[: SM.H].contiguous()
    kv2 = kv.squeeze(1)
    T = int(q.shape[0])
    lay: Optional[_Layer] = getattr(_CTX, "layer", None)
    row = q.stride(0) * q.element_size()
    off = (q.data_ptr() - lay.q.data_ptr()) // row if lay is not None else -1
    if (lay is not None and q.untyped_storage().data_ptr() == lay.q.untyped_storage().data_ptr()
            and 0 <= off and off + T <= lay.mxd.shape[0] and (q.data_ptr() - lay.q.data_ptr()) % row == 0):
        # the chunk's bf16 rows stay unwritten: vLLM's are scratch, read only by the quant pass
        # below, which skips the rows hk wrote
        SM.prefill_mxfp8(q, kv2, ind, ptr, sink, float(scale), lay.positions[off:off + T], lay.cos_sin,
                         lay.mxd[off:off + T], lay.mxs[off:off + T])
        lay.done.append((off, off + T))
        PATCH.count("hk_mxfp8")
        return None
    SM.prefill(q, kv2, ind, ptr, sink, float(scale), output)
    PATCH.count("hk")
    return None


def rocm_inverse_rope_mxfp8_rows(o, positions, cos_sin_cache, rope_head_dim, out_data, out_scale):
    """Stands in for vLLM's rocm_inverse_rope_mxfp8_rows: the rows no hk call wrote."""
    orig = _ORIG["quant"]
    lay: Optional[_Layer] = getattr(_CTX, "layer", None)
    if lay is None or not lay.done or out_data.data_ptr() != lay.mxd.data_ptr():
        return orig(o, positions, cos_sin_cache, rope_head_dim, out_data, out_scale)
    n = int(o.shape[0])
    cur = 0
    for lo, hi in sorted(lay.done) + [(n, n)]:
        if lo > cur:
            orig(o[cur:lo], positions[cur:lo], cos_sin_cache, rope_head_dim, out_data[cur:lo],
                 out_scale[cur:lo])
        cur = max(cur, hi)
    return None


def _mx_layout_ok(lay: "_Layer") -> bool:
    """The layer's MXFP8 destination is the layout prefill_mxfp8 writes: [T, 32 * 512] e4m3 and
    [T, 512] uint8 rows, contiguous; positions int64; cos/sin [P, 64] fp32. Anything else: vLLM's pass."""
    import torch  # noqa: PLC0415

    from ..ops import sparse_mla as SM  # noqa: PLC0415

    d, sc, p, cs = lay.mxd, lay.mxs, lay.positions, lay.cos_sin
    n = d.shape[0] if d.dim() == 2 else -1
    return (d.dtype == torch.float8_e4m3fn and d.dim() == 2 and d.shape[1] == SM.H * SM.D and d.is_contiguous()
            and sc.dtype == torch.uint8 and tuple(sc.shape) == (n, SM.H * SM.D // 32) and sc.is_contiguous()
            and p.dtype == torch.int64 and p.dim() == 1 and p.shape[0] == n and p.is_contiguous()
            and cs.dtype == torch.float32 and cs.dim() == 2 and cs.shape[1] == SM.D - SM.NOPE)


def _forward_prefill(self, q, positions, compressed_k_cache, swa_k_cache, output, attn_metadata,
                     swa_metadata, mxfp8_out=None):
    """Stands in for DeepseekV41ROCMAiterMLAAttention._forward_prefill: vLLM's, with the layer's MXFP8
    destination noted for the chunk calls."""
    orig = _ORIG["forward_prefill"]
    if (not ENABLED or not FUSE or mxfp8_out is None or getattr(self, "_use_aiter_sparse_mla", False)
            or getattr(_CTX, "layer", None) is not None):
        return orig(self, q, positions, compressed_k_cache, swa_k_cache, output, attn_metadata, swa_metadata,
                    mxfp8_out=mxfp8_out)
    try:
        n = int(swa_metadata.num_prefill_tokens)
        cs = _cos_sin_fp32(self.rotary_emb.cos_sin_cache)
        lay = _Layer(q, positions[:n], cs, mxfp8_out[0][:n], mxfp8_out[1][:n])
        if not _mx_layout_ok(lay):
            raise ValueError("not the MXFP8 layout hk writes")
    except Exception:  # noqa: BLE001 -- anything unexpected: vLLM's own path
        return orig(self, q, positions, compressed_k_cache, swa_k_cache, output, attn_metadata, swa_metadata,
                    mxfp8_out=mxfp8_out)
    _CTX.layer = lay
    try:
        return orig(self, q, positions, compressed_k_cache, swa_k_cache, output, attn_metadata, swa_metadata,
                    mxfp8_out=mxfp8_out)
    finally:
        _CTX.layer = None


#: The parameters of vLLM's entry points the stand-ins take, in order.
WANT = {"prefill": ["q", "kv", "indices", "topk_length", "scale", "head_dim", "nope_head_dim",
                    "rope_head_dim", "attn_sink", "output", "ragged_indices", "ragged_indptr"],
        "quant": ["o", "positions", "cos_sin_cache", "rope_head_dim", "out_data", "out_scale"],
        "forward_prefill": ["self", "q", "positions", "compressed_k_cache", "swa_k_cache", "output",
                            "attn_metadata", "swa_metadata", "mxfp8_out"]}


def _entry_points() -> Dict[str, Callable]:
    import importlib  # noqa: PLC0415

    ops = importlib.import_module(_OPS_MOD)
    rocm = importlib.import_module(_ROCM_MOD)
    return {"prefill": ops.rocm_sparse_attn_prefill, "quant": ops.rocm_inverse_rope_mxfp8_rows,
            "forward_prefill": rocm.DeepseekV41ROCMAiterMLAAttention._forward_prefill}


def problems() -> List[str]:
    """The engine contract (hk.integration.contract): vLLM's entry points with the parameters the
    stand-ins take, and the ragged-row builder they call. [] holds. Raises if a module moved."""
    import importlib  # noqa: PLC0415
    import inspect  # noqa: PLC0415

    have = _ORIG or _entry_points()
    out = [f"vLLM's {k} entry point takes {list(inspect.signature(f).parameters)}, not {WANT[k]}"
           for k, f in have.items() if list(inspect.signature(f).parameters) != WANT[k]]
    if not hasattr(importlib.import_module(_OPS_MOD), "build_ragged_indices_from_dense"):
        out.append("vLLM has no build_ragged_indices_from_dense")
    return out


def apply_patch() -> str:
    """Wrap vLLM's sparse prefill, its MXFP8 pass and _forward_prefill. Returns what happened."""
    from . import _common  # noqa: PLC0415

    if _ORIG:
        return "sparse_mla: already patched"
    from ..target import current_arch  # noqa: PLC0415

    if current_arch() != "gfx950":
        return f"sparse_mla: not patched -- gfx950's DMA gather and MFMA, not {current_arch()}"
    try:
        have = _entry_points()
    except Exception as e:  # noqa: BLE001
        return f"sparse_mla: not patched -- {type(e).__name__}: {e}"
    bad = problems()
    if bad:
        return f"sparse_mla: not patched -- {'; '.join(bad)}"
    import importlib  # noqa: PLC0415

    cls = importlib.import_module(_ROCM_MOD).DeepseekV41ROCMAiterMLAAttention
    fp = have["forward_prefill"]
    _ORIG.update(have)
    rocm_sparse_attn_prefill.__wrapped__ = have["prefill"]
    rocm_inverse_rope_mxfp8_rows.__wrapped__ = have["quant"]
    _forward_prefill.__wrapped__ = fp
    names = (_common.rebind(("vllm",), have["prefill"], rocm_sparse_attn_prefill)
             + _common.rebind(("vllm",), have["quant"], rocm_inverse_rope_mxfp8_rows))
    cls._forward_prefill = _forward_prefill
    return (f"sparse_mla -> hk for DeepSeek-V4.1 sparse MLA prefill calls of T >= {MIN_T}"
            f"{', MXFP8 rows in the kernel' if FUSE else ''} ({len(names)} bindings rebound)")


def revert() -> str:
    from . import _common  # noqa: PLC0415

    if not _ORIG:
        return "sparse_mla: not patched"
    import importlib  # noqa: PLC0415

    rocm = importlib.import_module(_ROCM_MOD)
    _common.rebind(("vllm",), rocm_sparse_attn_prefill, _ORIG["prefill"])
    _common.rebind(("vllm",), rocm_inverse_rope_mxfp8_rows, _ORIG["quant"])
    rocm.DeepseekV41ROCMAiterMLAAttention._forward_prefill = _ORIG["forward_prefill"]
    _ORIG.clear()
    return "sparse_mla -> vLLM"


def status() -> str:
    return "not patched" if not _ORIG else PATCH.summary()
