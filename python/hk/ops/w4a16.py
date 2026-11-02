"""4-bit weights, bf16 activations: the decode GEMM of an AWQ or GPTQ model.

At decode M is the batch and the cost is reading W, so 4-bit codes read a
quarter of what bf16 weights do. This is `decode_gemm` with W dequantized
on its way into registers (`hk.load_w4`): one 16-row block of W per
workgroup, its K split across the warps, the partial sums meeting in LDS.

**The layout** is hk's own, written once when the checkpoint is loaded.
Codes are packed eight to an int32 in K order, and the words interleaved by
16-row block and 32-column chunk, so that one 16-byte load per lane reads 16
rows' codes as 256 contiguous bytes. Row-major, the same load touched 16
lines 2 KB apart, and the kernel streamed codes at 223-328 GB/s. Per group
of `group` columns it keeps a scale s and the zero point times the scale,
zs, so a code dequantizes as q * s - zs, one FMA:

    qw  int32 [N, K / 8], ordered [N/16][K/32][16][4]: nibble t of word w of
        row n is column 8w + t
    s   bf16  [N, K / group]
    zs  bf16  [N, K / group]

`repack_awq` and `repack_gptq` write it from either checkpoint format.

**Symmetric.** A checkpoint with one zero point for every group, GPTQ's
uint4b8 (8), need not keep zs: `repack_gptq(..., symmetric=True)` returns zs
None and s in 16-row blocks,

    s   [N/16, K/group * 16]: block b, group g, row r at [b][16 g + r]

so the 16 rows a k-step reads take 32 contiguous bytes. The kernels compute
(q - 8) * s (`hk.load_w4(..., zero=8, s_blocked=True)`), in packed fp16 for
fp16 activations: q - 8 exact, the product rounded once to nearest even.
12.5% fewer bytes at GPTQ's 32-column groups, and half the decode work.
"""

from typing import Any, Dict, Optional, Tuple

from .. import dsl as hk
from ..types import bf16, fp16, fp32, i32
from . import _arch

_DTYPES = {"bf16": bf16, "fp16": fp16}

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


def _interleave(words):
    """Row-major words [N, K/8] -> [N/16][K/32][16][4], kept as [N, K/8]."""
    n, w = words.shape
    return words.reshape(n // 16, 16, w // 4, 4).permute(0, 2, 1, 3).reshape(n, w).contiguous()


def _deinterleave(qw):
    n, w = qw.shape
    return qw.reshape(n // 16, w // 4, 16, 4).permute(0, 2, 1, 3).reshape(n, w)


def _finish(q_nk, z_gn, s_gn, group: int, dtype):
    """Codes [N, K] and per-group zeros and scales [K/group, N] -> hk's layout,
    the scales in `dtype` (the activations' type; None: the checkpoint's
    scales' if 16-bit, else bf16)."""
    import torch  # noqa: PLC0415

    n, k = q_nk.shape
    if n % 16 or k % 32:
        raise ValueError(f"hk's 4-bit layout takes N a multiple of 16 and K of 32, "
                         f"got {n} x {k}")
    if dtype is None:
        dtype = s_gn.dtype if s_gn.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    s = s_gn.t().float()
    zs = z_gn.t().float() * s
    return _interleave(_pack_k(q_nk)), s.to(dtype).contiguous(), zs.to(dtype).contiguous()


def repack_awq(qweight, qzeros, scales, group: int, dtype=None):
    """An AutoAWQ GEMM checkpoint -- qweight int32 [K, N/8], qzeros int32
    [K/group, N/8], both packed along N in AWQ's order, scales [K/group, N] --
    as (qw, s, zs) in hk's layout."""
    q = _unpack_cols(qweight, AWQ_ORDER)            # [K, N]
    z = _unpack_cols(qzeros, AWQ_ORDER)             # [K/group, N]
    return _finish(q.t().contiguous(), z, scales, group, dtype)


#: The zero point of a symmetric 4-bit checkpoint (GPTQ's uint4b8).
SYMMETRIC_ZERO = 8


def repack_gptq(qweight, qzeros, scales, group: int, *, zero_offset: int = 1,
                dtype=None, symmetric: bool = False):
    """A GPTQ checkpoint without act-order -- qweight int32 [K/8, N] packed
    along K, qzeros int32 [K/group, N/8] packed along N, scales [K/group, N]
    -- as (qw, s, zs). `zero_offset` 1 is GPTQ v1's stored zero, one short.

    `symmetric`: (qw, s, None) for the kernels' symmetric form, which takes
    every zero point to be 8. Refused (ValueError) unless every stored zero
    point is: a checkpoint is checked, not assumed."""
    import torch  # noqa: PLC0415

    k8, n = qweight.shape
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    q = ((qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(k8 * 8, n)
    z = _unpack_cols(qzeros).to(torch.int32) + zero_offset
    if symmetric:
        if not bool((z == SYMMETRIC_ZERO).all()):
            raise ValueError(f"repack_gptq(symmetric=True): the checkpoint's zero points are not "
                             f"all {SYMMETRIC_ZERO} (with zero_offset {zero_offset})")
        qw, s, _ = _finish(q.t().contiguous().to(torch.uint8), z, scales, group, dtype)
        return qw, block_scales(s), None
    return _finish(q.t().contiguous().to(torch.uint8), z, scales, group, dtype)


def block_scales(s):
    """Scales [N, G] -> the symmetric form's 16-row blocks [N/16, G * 16]."""
    n, g = s.shape
    return s.reshape(n // 16, 16, g).permute(0, 2, 1).reshape(n // 16, g * 16).contiguous()


def unblock_scales(s):
    """`block_scales`' inverse: [N/16, G * 16] -> [N, G]."""
    b, g16 = s.shape
    return s.reshape(b, g16 // 16, 16).permute(0, 2, 1).reshape(b * 16, g16 // 16)


def dequant(qw, s, zs, group: int):
    """hk's layout back to a bf16 [N, K]: the reference, and the path for
    batches the decode kernel does not take. zs None: symmetric -- (q - 8) * s,
    s in 16-row blocks."""
    q = _unpack_cols(_deinterleave(qw)).float()     # [N, K]
    if zs is None:
        s = unblock_scales(s)
    sc = s.float().repeat_interleave(group, dim=1)
    z = sc * SYMMETRIC_ZERO if zs is None else zs.float().repeat_interleave(group, dim=1)
    return (q * sc - z).to(s.dtype)


def w4a16_kernel(name: str, *, tokens: int = 1, slices: int = 4, warps: int = 16,
                 group: int = 128, blocks: int = 1, dtype=bf16, symmetric: bool = False):
    """`tokens` 16-row token tiles (M <= 16 * tokens), `slices` 16-deep K
    slices per load, `warps` waves sharing each workgroup's `blocks` 16-row
    blocks of W, `group` columns per scale. K is a multiple of
    16 * slices * warps, and of group; group is a multiple of 16 * slices,
    so a load's codes share one scale.

    `blocks` > 1 reads each token tile once for several blocks of W: at four
    bits a 16 x 64 block of W is 512 bytes and the tokens' tile beside it
    2 KB, so the token reads, not the weights, can be what fills the L2.

    `symmetric`: the kernel takes (x, qw, s, c), every zero point 8, no zs, s
    in 16-row blocks (module docstring)."""
    m_rows = 16 * tokens
    kc = 16 * slices
    rows = 16 * blocks
    if group % kc:
        raise ValueError(f"group {group} must be a multiple of the {kc}-column load")
    ty_w = hk.RT(dtype, rows, kc, "row")
    ty_x = hk.RT(dtype, m_rows, kc, "row")
    ty_acc = hk.RT(fp32, rows, m_rows, "col")    # C^T: W rows x tokens
    ty_out = hk.RT(fp32, m_rows, rows, "row")
    st_part = hk.ST(fp32, m_rows, rows)
    in_flight = max(2, 8 // (tokens * blocks))

    @hk.func
    def gemm(x, qw, s, zs, c):
        n0 = hk.arch.block_idx.x * rows
        wid = hk.arch.warp_id()
        acc = hk.zeros(ty_acc)
        for t in hk.range(x.cols // (kc * warps)):
            k = (t * warps + wid) * kc
            w_t = (hk.load_w4(ty_w, qw, s, None, n0, k, group=group, zero=SYMMETRIC_ZERO,
                              s_blocked=True)
                   if zs is None else hk.load_w4(ty_w, qw, s, zs, n0, k, group=group))
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

    if symmetric:
        def body(x: hk.Tensor[dtype], qw: hk.Tensor[i32], s: hk.Tensor[dtype],
                 c: hk.Tensor[dtype]):
            gemm(x, qw, s, None, c)
    else:
        def body(x: hk.Tensor[dtype], qw: hk.Tensor[i32], s: hk.Tensor[dtype],
                 zs: hk.Tensor[dtype], c: hk.Tensor[dtype]):
            gemm(x, qw, s, zs, c)

    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name,
                     grid=lambda p: (p.qw.rows // rows, 1, 1))


#: Per token-tile count: decode_gemm's measured schedules (chapter 9), with
#: the load kept at a whole group's divisor.
SCHEDULES: Dict[int, Dict[str, int]] = {
    1: dict(slices=4, warps=16, blocks=1),
    2: dict(slices=2, warps=16, blocks=1),
    4: dict(slices=2, warps=8, blocks=1),
}
MAX_TOKENS = max(SCHEDULES)

#: One token tile at K up to this: two 16-row blocks of W a workgroup on 8
#: warps, 5-22% faster than one on 16 at Qwen3-8B's 4096-deep layers
#: (tools/hk-bench/w4a16_ab.py) and 6% slower at its 12288-deep one.
SHALLOW_K = 4096


def schedule_for(m: int, k: int, group: int = 128) -> Optional[Dict[str, int]]:
    """The schedule for M rows at depth K. A load's 16 x slices columns share
    one scale, so a group narrower than the load (GPTQ's 32) narrows it."""
    t = tokens_for(m)
    if t is None:
        return None
    s = dict(slices=4, warps=8, blocks=2) if t == 1 and k <= SHALLOW_K else dict(SCHEDULES[t])
    if 16 * s["slices"] > group and group % 16 == 0:
        s["slices"] = group // 16
    return s

_KERNELS: Dict[Tuple[Any, ...], Any] = {}
_OPS: Dict[Tuple[Any, ...], Any] = {}


def tokens_for(m: int) -> Optional[int]:
    t = -(-m // 16)
    return next((k for k in sorted(SCHEDULES) if k >= t), None)


def kernel_for(tokens: int, slices: int, warps: int, group: int, blocks: int = 1,
               dtype: str = "bf16", symmetric: bool = False):
    key = (tokens, slices, warps, group, blocks, dtype, symmetric)
    if key not in _KERNELS:
        _KERNELS[key] = w4a16_kernel(
            f"w4a16_t{tokens}_s{slices}_w{warps}_g{group}_b{blocks}_{dtype}" + ("_sym" if symmetric else ""),
            tokens=tokens, slices=slices, warps=warps, group=group, blocks=blocks,
            dtype=_DTYPES[dtype], symmetric=symmetric)
    return _KERNELS[key]


def _op_for(*key):
    """The kernel's launch: its torch op outside torch.compile's tracing
    called without the dispatcher (`torch_ext.launcher`)."""
    if key not in _OPS:
        from ..runtime.torch_ext import launcher  # noqa: PLC0415

        _OPS[key] = launcher(kernel_for(*key))
    return _OPS[key]


def takes(m: int, n: int, k: int, group: int) -> bool:
    """Whether the decode kernel runs an (M, N, K) GEMM at this group size."""
    s = schedule_for(m, k, group)
    if s is None or k % group:
        return False
    kc = 16 * s["slices"]
    return (n % (16 * s["blocks"]) == 0 and group % kc == 0
            and k % (kc * s["warps"]) == 0)


#: (M, N, K, group, dtype) -> the launch that takes the call, or None for the
#: dequantize path: `takes`, `schedule_for` and the kernel's name, decided once.
_PLANS: Dict[Tuple[Any, ...], Any] = {}
#: Shapes memoized; past it a call decides afresh.
_PLAN_CAP = 4096


def plan(m: int, n: int, k: int, group: int, dtype, symmetric: bool = False):
    """The decode kernel's launch for this call -- (x, qw, s, zs, out), or (x, qw, s, out)
    `symmetric` -- or None where `matmul` takes the dequantize path: memoized `takes` and
    schedule."""
    key = (m, n, k, group, dtype, symmetric)
    try:
        return _PLANS[key]
    except KeyError:
        pass
    import torch  # noqa: PLC0415

    names = {torch.bfloat16: "bf16", torch.float16: "fp16"}
    run = None
    if dtype in names and takes(m, n, k, group):
        sch = schedule_for(m, k, group)
        run = _op_for(tokens_for(m), sch["slices"], sch["warps"], group, sch["blocks"],
                      names[dtype], symmetric)
    if len(_PLANS) < _PLAN_CAP:
        _PLANS[key] = run
    return run


def matmul(x, qw, s, zs, group: int, out=None):
    """x @ W.T for W in hk's 4-bit layout: the decode kernel up to 64 rows of
    x, dequantize-then-F.linear above (or for a shape it does not take).
    zs None: the symmetric form (every zero point 8; `repack_gptq(symmetric=True)`)."""
    import torch  # noqa: PLC0415

    _arch.require("hk.ops.w4a16.matmul")
    from ..runtime import hostpath  # noqa: PLC0415

    if not hostpath.ENABLED:
        return _matmul_legacy(x, qw, s, zs, group, out)
    m, k = x.shape
    n = qw.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=s.dtype)
    run = plan(m, n, k, group, x.dtype, zs is None) if x.dtype == s.dtype else None
    if run is not None:
        xc = x if x.is_contiguous() else x.contiguous()
        if zs is None:
            run(xc, qw, s, out)
        else:
            run(xc, qw, s, zs, out)
        return out
    out.copy_(torch.nn.functional.linear(x.to(s.dtype), dequant(qw, s, zs, group)))
    return out


def _op_for_legacy(*key):
    if key not in _OPS_LEGACY:
        from ..runtime.torch_ext import torch_op  # noqa: PLC0415

        _OPS_LEGACY[key] = torch_op(kernel_for(*key))
    return _OPS_LEGACY[key]


_OPS_LEGACY: Dict[Tuple[Any, ...], Any] = {}


def _matmul_legacy(x, qw, s, zs, group: int, out=None):
    """`matmul` before the host fast paths, line for line (HK_HOST_FASTPATH=0)."""
    import torch  # noqa: PLC0415

    m, k = x.shape
    n = qw.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=s.dtype)
    names = {torch.bfloat16: "bf16", torch.float16: "fp16"}
    if x.dtype == s.dtype and x.dtype in names and takes(m, n, k, group):
        t = tokens_for(m)
        sch = schedule_for(m, k, group)
        op = _op_for_legacy(t, sch["slices"], sch["warps"], group, sch["blocks"], names[x.dtype],
                            zs is None)
        xc = x if x.is_contiguous() else x.contiguous()
        if zs is None:
            op(xc, qw, s, out)
        else:
            op(xc, qw, s, zs, out)
        return out
    out.copy_(torch.nn.functional.linear(x.to(s.dtype), dequant(qw, s, zs, group)))
    return out


__all__ = ["AWQ_ORDER", "MAX_TOKENS", "SCHEDULES", "SHALLOW_K", "SYMMETRIC_ZERO", "block_scales",
           "dequant", "kernel_for", "unblock_scales",
           "matmul", "plan", "schedule_for",
           "repack_awq", "repack_gptq", "takes", "tokens_for", "w4a16_kernel"]
