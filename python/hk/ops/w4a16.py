"""4-bit weights, bf16 activations: the decode GEMM of an AWQ or GPTQ model.

At decode M is the batch and the cost is reading W, so 4-bit codes read a
quarter of what bf16 weights do. This is `decode_gemm` with W dequantized
on its way into registers (`hk.load_w4`): one 16-row block of W per
workgroup, its K split across the warps, the partial sums meeting in LDS.

**The layout** is hk's own, written once when the checkpoint is loaded.
A row of W (an output feature) keeps its K codes together, eight to an int32
in K order, so a lane reads its row's codes for a 16-column base tile as one
8-byte load. Per group of `group` columns it keeps a scale s and the zero
point times the scale, zs, so a code dequantizes as q * s - zs, one FMA:

    qw  int32 [N, K / 8]      nibble t of qw[n][w] is column 8w + t
    s   bf16  [N, K / group]
    zs  bf16  [N, K / group]

`repack_awq` and `repack_gptq` write it from either checkpoint format.
"""

from typing import Any, Dict, Optional, Tuple

from .. import dsl as hk
from ..types import bf16, fp32, i32

#: AWQ packs eight output columns to an int32, column c in nibble AWQ_ORDER[c].
AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _unpack_cols(packed, order=None):
    """int32 [R, C/8] -> uint8 [R, C], column 8j + c from nibble order[c] of
    word j (natural order when None)."""
    import torch  # noqa: PLC0415

    order = order or tuple(range(8))
    shifts = torch.tensor([4 * o for o in order], device=packed.device, dtype=torch.int32)
    q = (packed.unsqueeze(-1) >> shifts) & 0xF
    return q.reshape(packed.shape[0], -1).to(torch.uint8)


def _pack_k(q):
    """Codes [N, K] -> int32 [N, K/8], nibble t of word w being column 8w + t."""
    import torch  # noqa: PLC0415

    n, k = q.shape
    shifts = torch.arange(0, 32, 4, device=q.device, dtype=torch.int64)
    words = (q.to(torch.int64).reshape(n, k // 8, 8) << shifts).sum(dim=-1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words)
    return words.to(torch.int32).contiguous()


def _finish(q_nk, z_gn, s_gn, group: int):
    """Codes [N, K] and per-group zeros and scales [K/group, N] -> hk's layout."""
    import torch  # noqa: PLC0415

    s = s_gn.t().float()
    zs = z_gn.t().float() * s
    return _pack_k(q_nk), s.to(torch.bfloat16).contiguous(), zs.to(torch.bfloat16).contiguous()


def repack_awq(qweight, qzeros, scales, group: int):
    """An AutoAWQ GEMM checkpoint -- qweight int32 [K, N/8], qzeros int32
    [K/group, N/8], both packed along N in AWQ's order, scales [K/group, N] --
    as (qw, s, zs) in hk's layout."""
    q = _unpack_cols(qweight, AWQ_ORDER)            # [K, N]
    z = _unpack_cols(qzeros, AWQ_ORDER)             # [K/group, N]
    return _finish(q.t().contiguous(), z, scales, group)


def repack_gptq(qweight, qzeros, scales, group: int, *, zero_offset: int = 1):
    """A GPTQ checkpoint without act-order -- qweight int32 [K/8, N] packed
    along K, qzeros int32 [K/group, N/8] packed along N, scales [K/group, N]
    -- as (qw, s, zs). `zero_offset` 1 is GPTQ v1's stored zero, one short."""
    import torch  # noqa: PLC0415

    k8, n = qweight.shape
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    q = ((qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(k8 * 8, n)
    z = _unpack_cols(qzeros).to(torch.int32) + zero_offset
    return _finish(q.t().contiguous().to(torch.uint8), z, scales, group)


def dequant(qw, s, zs, group: int):
    """hk's layout back to a bf16 [N, K]: the reference, and the path for
    batches the decode kernel does not take."""
    q = _unpack_cols(qw).float()                    # [N, K]
    sc = s.float().repeat_interleave(group, dim=1)
    z = zs.float().repeat_interleave(group, dim=1)
    return (q * sc - z).to(s.dtype)


def w4a16_kernel(name: str, *, tokens: int = 1, slices: int = 4, warps: int = 16,
                 group: int = 128):
    """`tokens` 16-row token tiles (M <= 16 * tokens), `slices` 16-deep K
    slices per load, `warps` waves sharing each 16-row block of W, `group`
    columns per scale. K is a multiple of 16 * slices * warps, and of group;
    group is a multiple of 16 * slices, so a load's codes share one scale."""
    m_rows = 16 * tokens
    kc = 16 * slices
    if group % kc:
        raise ValueError(f"group {group} must be a multiple of the {kc}-column load")
    ty_w = hk.RT(bf16, 16, kc, "row")
    ty_x = hk.RT(bf16, m_rows, kc, "row")
    ty_acc = hk.RT(fp32, 16, m_rows, "col")      # C^T: W rows x tokens
    ty_out = hk.RT(fp32, m_rows, 16, "row")
    st_part = hk.ST(fp32, m_rows, 16)
    in_flight = max(2, 8 // tokens)

    def body(x: hk.Tensor[bf16], qw: hk.Tensor[i32], s: hk.Tensor[bf16],
             zs: hk.Tensor[bf16], c: hk.Tensor[bf16]):
        n0 = hk.arch.block_idx.x * 16
        wid = hk.arch.warp_id()
        acc = hk.zeros(ty_acc)
        for t in hk.range(x.cols // (kc * warps)):
            k = (t * warps + wid) * kc
            w_t = hk.load_w4(ty_w, qw, s, zs, n0, k, group=group)
            x_t = x[hk.elem_coord(0, 0, 0, k)].load(ty_x, bounded=True)
            acc += w_t @ x_t.T
        out = hk.transpose(acc)
        if warps > 1:
            parts = hk.lds.alloc(st_part, count=warps, name="parts")
            parts[wid].store(out)
            hk.arch.barrier(drain=True)
        if wid == 0:
            if warps > 1:
                for i in range(1, warps):
                    if i > 1 and (i - 1) % in_flight == 0:
                        hk.arch.sched_barrier()
                    out += parts[i].load(ty_out)
            c[hk.elem_coord(0, 0, 0, n0)].store(out, bounded=True)

    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: (p.qw.rows // 16, 1, 1))


#: Per token-tile count: decode_gemm's measured schedules (chapter 9), with
#: the load kept at a whole group's divisor.
SCHEDULES: Dict[int, Dict[str, int]] = {
    1: dict(slices=4, warps=16),
    2: dict(slices=2, warps=16),
    4: dict(slices=2, warps=8),
}
MAX_TOKENS = max(SCHEDULES)

_KERNELS: Dict[Tuple[Any, ...], Any] = {}
_OPS: Dict[Tuple[Any, ...], Any] = {}


def tokens_for(m: int) -> Optional[int]:
    t = -(-m // 16)
    return next((k for k in sorted(SCHEDULES) if k >= t), None)


def kernel_for(tokens: int, slices: int, warps: int, group: int):
    key = (tokens, slices, warps, group)
    if key not in _KERNELS:
        _KERNELS[key] = w4a16_kernel(
            f"w4a16_t{tokens}_s{slices}_w{warps}_g{group}", tokens=tokens,
            slices=slices, warps=warps, group=group)
    return _KERNELS[key]


def _op_for(*key):
    if key not in _OPS:
        from ..runtime.torch_ext import torch_op  # noqa: PLC0415

        _OPS[key] = torch_op(kernel_for(*key))
    return _OPS[key]


def takes(m: int, n: int, k: int, group: int) -> bool:
    """Whether the decode kernel runs an (M, N, K) GEMM at this group size."""
    t = tokens_for(m)
    if t is None or n % 16 or k % group:
        return False
    s = SCHEDULES[t]
    kc = 16 * s["slices"]
    return group % kc == 0 and k % (kc * s["warps"]) == 0


def matmul(x, qw, s, zs, group: int, out=None):
    """x @ W.T for W in hk's 4-bit layout: the decode kernel up to 64 rows of
    x, dequantize-then-F.linear above (or for a shape it does not take)."""
    import torch  # noqa: PLC0415

    m, k = x.shape
    n = qw.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    if x.dtype == torch.bfloat16 and takes(m, n, k, group):
        t = tokens_for(m)
        sch = SCHEDULES[t]
        _op_for(t, sch["slices"], sch["warps"], group)(
            x if x.is_contiguous() else x.contiguous(), qw, s, zs, out)
        return out
    out.copy_(torch.nn.functional.linear(x.to(torch.bfloat16), dequant(qw, s, zs, group)))
    return out


__all__ = ["AWQ_ORDER", "MAX_TOKENS", "SCHEDULES", "dequant", "kernel_for", "matmul",
           "repack_awq", "repack_gptq", "takes", "tokens_for", "w4a16_kernel"]
