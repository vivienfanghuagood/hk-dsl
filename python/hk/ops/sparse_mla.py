"""DeepSeek-V4.1's sparse MLA prefill attention on gfx950 (CDNA4, MI355X), written in the DSL.

The op vLLM's ROCm path runs for a prefill chunk (`rocm_sparse_attn_prefill`, AITER's opus kernel at
T >= 1024 queries), on the same inputs:

  * q [T, 32, 512] bf16 -- a TP2 rank's 32 heads; the row is 448 nope + 64 rope dims;
  * kv [R, 512] bf16 -- the rows vLLM gathered out of its fp8 caches; V is K, the whole row;
  * indices / indptr int32 -- per query, the rows it attends (CSR): its sliding window and the
    indexer's top-k; indices may run past indptr[T] (vLLM over-allocates);
  * attn_sink [32] fp32 -- exp(sink) joins each head's softmax denominator;
  * out [T, 32, 512] bf16.

`prefill_mxfp8` writes instead what the layer hands to wo_a: the rows inverse-RoPE'd on lanes
448-511 (GPT-J pairs) and MXFP8-quantized, one E8M0 scale a 32 lanes -- the arithmetic of vLLM's
`_inverse_rope_mxfp8_quant_kernel` on the bf16 rows `prefill` writes, bit for bit -- and no bf16
rows (vLLM's are a scratch buffer only that pass reads).

The kernel. One query a workgroup of four warps, all 32 heads; two workgroups a CU (71 KB of LDS,
224 VGPRs), so one workgroup's MFMAs fill the other's waits at its per-block exchanges -- the same
partition as one eight-warp workgroup a CU measured 1.25x slower (it waited at all its own barriers).
The query's rows are walked in 32-row blocks, gathered by index into LDS by gfx950's DMA
(`g.load(gather=)`), double-buffered. Warp w = (r = w % 2, dh = w // 2):

  QK  Sᵀ (kv rows 16r.. x 32 heads) over d in [256 dh, 256 dh + 256) = K · Qᵀ (16x16x32, Q in
      registers); warp r + 2's half goes through LDS to warp r, which finishes S;
  softmax: the block's per-head max over the two row groups (LDS); lazy online softmax -- the
      running max moves only past a step of 8 (log2), so O is rescaled only then; l sums the
      bf16 P that P·V uses, so numerator and denominator round alike;
  PV  split by D -- warp w: Oᵀ (d 128w.. x 32 heads) += Vᵀ · Pᵀ (32x32x16). V is the LDS tile K came
      from: V's col fragment (16 kv x 32 d, a run of 8) and K's row fragment (16 x 32, run 8)
      read the same st_16x32 layout.

The sink and 1/l at the end; Oᵀ stored transposed. The MXFP8 epilogue works on O where it sits:
Oᵀ taken as a strip (`lane_strip`), a 32-lane MX group is a lane's run of a 32x32 fragment and lane
l ^ 32's (`group_max(32, xor=(32,))`), a rope pair's partner the next element of the run
(`run_xor(1)`); lanes l and l ^ 32 then trade half their values (`lane_gather`) so each stores 16
consecutive fp8 of its head at once.

Measured on an MI355X against the kernels on vLLM's path, DSV4.1-Flash TP2 calls captured from a
real-weight run: top-k layers at T 8192, 640 rows a query, 1.11-1.13x opus
(attention); with the MXFP8 rows 1.22x AITER's fused fp8 sparse_mla_fwd and 1.39x opus + vLLM's
quant; sliding-window layers 1.15-1.20x the faster of opus and AITER (attention), 1.12x and 1.97x
with the MXFP8 rows. At T <= 256 one query a workgroup leaves the machine half empty and opus is
faster: `MIN_T`.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from .. import dsl as hk
from ..types import bf16, fp32, fp8e4m3, i8, i32

NEG_INF = float("-inf")
LOG2E = math.log2(math.e)

#: Query heads (a TP2 rank of DeepSeek-V4.1-Flash), the row (448 nope + 64 rope).
H, D, NOPE = 32, 512, 448
#: KV rows a block; a QK warp's rows; the D a QK warp takes; a PV warp's D.
KB, KQ, DH = 32, 16, 256
NKV = KB // KQ
WARPS = 2 * NKV
DW = D // WARPS
#: The running max moves only when a head's block max passes it by more than this (log2 units).
LAZY = 8.0
#: The fewest queries a call should have for this kernel: below it a call holds too few
#: workgroups for the machine (one a query); measured opus 27-33 us against 41-42 at T 128-256.
MIN_T = 1024

T_Q = hk.RT(bf16, H, DH, "row")                    # Q, all heads, this warp's D half
T_K = hk.RT(bf16, KQ, DH, "row")                   # this warp's K rows, its D half
T_S = hk.RT(fp32, KQ, H, "col")                    # Sᵀ [kv, head]
T_PB = hk.RT(bf16, KQ, H, "col")                   # P, as the softmax makes it
T_V = hk.RT(bf16, KB, DW, "col", frag=(16, 32))    # V: every row of the block x this warp's 128 d
T_P = hk.RT(bf16, KB, H, "col", frag=(16, 32))     # Pᵀ [kv, head]: the PV B operand
T_O = hk.RT(fp32, DW, H, "col", frag=(32, 32))     # Oᵀ [d, head]
V_S = hk.row_vec(T_S)                              # one value a head, in S's layout
V_O = hk.row_vec(T_O)                              # ... in O's
ST_KV = hk.ST(bf16, KB, D)
ST_P = hk.ST(bf16, KB, H)
ST_SX = hk.ST(fp32, KQ, H)                         # a D-half-1 warp's partial Sᵀ, for its partner

#: vLLM's _mxfp8_scale_bits, as written there: the biased E8M0 exponent that puts amax at the top
#: of e4m3's range. (From amax's bits -- hk.mx_quant_store's rule -- differs from it in about one
#: scale in four million: Triton's log2 just past 1.75 x 2^e.) Of an hk.map element.
def _bits(a):
    return hk.min(hk.max(hk.math.ceil(hk.math.log2(hk.max(a, 1.1754943508222875e-38) / 448.0))
                         + 127.0, 0.0), 254.0)


ST16 = hk.RS(fp32, 1, 64 * 64, 16)     # Oᵀ's registers, a run a 32x32 fragment


def _lane(c):
    """The lane of a run-16 strip element at strip column c."""
    return (c >> 4) & 63


def _dl(c):
    """d within the warp's 128 of a run-16 strip element at strip column c (lane l = (c >> 4) & 63,
    e = c & 15)."""
    return ((c >> 10) << 5) + (((c & 15) >> 2) << 3) + ((_lane(c) >> 5) << 2) + (c & 3)


def _qq(c):
    """The 16-byte slot a run-16 element's quarter q = (c & 15) >> 2 goes to: q in half 0 of the
    lanes, 3 - q in half 1."""
    return hk.where(_lane(c) >= 32, 3 - ((c & 15) >> 2), (c & 15) >> 2)


def _slot(own, y4, y8, c):
    """What slot _qq(c) of a 16-byte store takes: own (slots 0 and 2), the partner's e ^ 4 (1),
    its e ^ 8 (3). Each comparison takes its own _qq, as the C++ this replaced did: with one
    shared, clang schedules the lane exchange around it differently (V1, docs/design/dsl-surface.md
    §7)."""
    return hk.where((_qq(c) == 0) | (_qq(c) == 2), own, hk.where(_qq(c) == 1, y4, y8))


def _quant_store(r, mxd, mxs, t, w):
    """The MXFP8 rows of r (a run-16 strip of Oᵀ, bf16 values) to mxd / mxs row t (a query's 32 x 512)."""
    b = hk.map(_bits, hk.map(abs, r).group_max(32, xor=(32,)))
    qv = hk.map(lambda x, e: x * hk.math.exp2(127.0 - e), r, b)
    c = hk.iota(ST16)
    # one scale a 32-d group: element 0 of lane l < 32's run k is d 32k of head l
    mxs.store_elems(b, hk.map(lambda c, w: hk.where(((c & 15) == 0) & (_lane(c) < 32),
                                                    (_lane(c) & 31) * (D // 32) + DW // 32 * w + (c >> 10),
                                                    -1), c, w), row=t)
    # 16 bytes a store: lane l (half h = l >> 5) and lane l ^ 32 trade half their values so each holds
    # d 32k + 16h .. +15 of its head. h 0 wants [own q0, its q0, own q1, its q1], h 1 [its q2, own q2,
    # its q3, own q3] (q: a run of 4; the partner's q0 is d 4-7 for h 0, ...)
    # slot QQ (h 0: q, h 1: 3 - q) takes: 0 own e, 1 the partner's e ^ 4, 2 own e ^ 12, 3 the partner's e ^ 8
    own = hk.map(lambda x, x12, c: hk.where(_qq(c) == 2, x12, x), qv, qv.run_xor(12), c)
    y = qv.lane_gather(_lane(c) ^ 32)
    rv = hk.map(_slot, own, y.run_xor(4), y.run_xor(8), c)
    # run-16 chunk k of lane l: d 32k + 16h
    mxd.store_elems(rv, hk.map(lambda c, w: (_lane(c) & 31) * D + DW * w + ((c >> 10) << 5)
                               + ((_lane(c) >> 5) << 4), c, w), runs=True, row=t)


def kernel(name: str, *, scale: float, mxfp8: bool, arch=None):
    """The kernel: bf16 rows, and with `mxfp8` also the MXFP8 rows (module docstring)."""
    scale_l2e = float(scale) * LOG2E

    def body(q: hk.Tensor[bf16], kv: hk.Tensor[bf16], ind: hk.Tensor[i32], ptr: hk.Tensor[i32],
             sink: hk.Tensor[fp32], o: hk.Tensor[bf16], pos: hk.Tensor[i32], cs: hk.Tensor[fp32],
             mxd: hk.Tensor[fp8e4m3], mxs: hk.Tensor[i8]):
        t = hk.arch.block_idx.x
        w = hk.arch.warp_id()
        r = w % NKV
        dh = w // NKV
        g = hk.group(WARPS)
        Ks = hk.lds.alloc(ST_KV, count=2, name="Ks")
        Ps = hk.lds.alloc(ST_P, name="Ps")
        Sx = hk.lds.alloc(ST_SX, count=NKV, name="Sx")
        # per-warp vectors through LDS, read back in S's layout and in O's (16x16 vs 32x32 fragments)
        xw = hk.lds.alloc_vec(hk.lds.shared_vec(V_S, WARPS), name="xw")
        xl = hk.lds.alloc_vec(hk.lds.shared_vec(V_S, WARPS), name="xl")

        a = ptr[hk.elem_coord(0, 0, 0, t)].load_scalar(invariant=True)
        n = ptr[hk.elem_coord(0, 0, 0, t + 1)].load_scalar(invariant=True) - a
        n_pair = hk.cdiv(n, 2 * KB)

        g.load(Ks[0], kv.at[0, 0], wait=False, gather=hk.RowIndex(ind, a), rows=n)
        qd = q[hk.elem_coord(0, 0, t * H, dh * DH)].load(T_Q)

        S = hk.zeros(T_S)
        Pb = hk.zeros(T_PB)
        O = hk.zeros(T_O)
        # the running max in both layouts: finite, so the first step up is finite (-inf + inf would be
        # NaN); an empty query's sink term exp2(sink - m) is then inf and its output 0
        m = hk.full(V_S, -1e30)
        mo = hk.full(V_O, -1e30)
        bm = hk.zeros(V_S)
        bo = hk.zeros(V_O)
        sc = hk.zeros(V_S)
        sco = hk.zeros(V_O)
        ell = hk.zeros(V_S)

        for i in range(n_pair):
            for s in range(2):
                blk = 2 * i + s
                # rows past the query's read zero, their indices unread: the last query's list may
                # end its allocation (vLLM's packing leaves no slack when every row is full)
                g.load(Ks[1 - s], kv.at[0, 0], wait=False,
                       gather=hk.RowIndex(ind, a + (blk + 1) * KB), rows=n - (blk + 1) * KB)
                g.wait_for(Ks[s])
                hk.arch.barrier()
                # QK: this warp's 16 rows, its D half, every head; the halves summed through LDS
                kt = Ks[s].subtile(KQ, DH, r, dh).load(T_K)
                S.zero_()
                S += kt @ qd.T
                if dh == 1:
                    Sx[r].store(S)
                hk.arch.barrier(drain=True)
                if dh == 0:
                    S += Sx[r].load(T_S)
                S *= scale_l2e
                left = n - blk * KB - r * KQ
                if left < KQ:
                    S.lower_fill_(hk.max(left, 0), NEG_INF)
                # the block's per-head max over the warps that finished S, in both layouts
                xw.store(S.max(axis=0), w)
                hk.arch.barrier(drain=True)
                xw.load(V_S, 0).copy(into=bm)
                xw.load(V_O, 0).copy(into=bo)
                for u in range(1, NKV):
                    bm.max_(xw.load(V_S, u))
                    bo.max_(xw.load(V_O, u))
                bm.max_(m)
                bo.max_(mo)
                grow = bm.any_above(m, LAZY)     # one vote a wave; the inputs are equal in every warp
                bm.__isub__(m)
                bm.__imul__(grow)
                m.__iadd__(bm)
                hk.math.mul(bm, -1.0, into=sc)
                sc.exp2_()                       # 1 unless the max moved
                ell.__imul__(sc)
                if grow > 0:
                    bo.__isub__(mo)
                    mo.__iadd__(bo)
                    hk.math.mul(bo, -1.0, into=sco)
                    sco.exp2_()
                    O.__imul__(sco)
                S.__isub__(m)
                S.exp2_()
                S.to(bf16).copy(into=Pb)
                Pb.to(fp32).copy(into=S)
                ell.__iadd__(S.sum(axis=0))
                if dh == 0:                      # the pair's P is the same: one of them writes it
                    Ps.subtile(KQ, H, r, 0).store(Pb)
                hk.arch.barrier(drain=True)
                # PV: this warp's 128 columns of every row of the block
                pt = Ps.load(T_P)
                vt = Ks[s].subtile(KB, DW, 0, w).load(T_V)
                O += vt.T @ pt
                hk.arch.barrier()

        g.wait_for(Ks[0])
        hk.arch.barrier()
        # l over the row groups (the warps that finished S), in O's layout; the sink joins it
        xl.store(ell, w)
        hk.arch.barrier(drain=True)
        lt = xl.load(V_O, 0)
        for u in range(1, NKV):
            lt.__iadd__(xl.load(V_O, u))
        sk = sink[hk.elem_coord(0, 0, 0, 0)].load(V_O)
        sk *= LOG2E
        hk.math.sub(sk, mo, into=sk)
        sk.exp2_()
        lt.__iadd__(sk)
        inv = hk.full(V_O, 1.0)
        inv.__itruediv__(lt)
        O.__imul__(inv)
        if not mxfp8:
            o[hk.elem_coord(0, 0, t * H, w * DW)].store(hk.transpose(O))
        else:
            # Oᵀ [d 128w.., head] where it sits, a run-16 strip: lane l's run k is fragment k's 16 values,
            # d = 32k + 8(e >> 2) + 4(l >> 5) + (e & 3), head l & 31 (hk.lane_coords; written out below so
            # nothing but the values stays live). A 32-d MX group is a lane's run and lane l ^ 32's.
            xb = hk.map(lambda x: x.to(bf16).to(fp32), O.lane_strip(16))   # the bf16 rows vLLM's pass reads
            if w == WARPS - 1:
                # d 384..511: 448.. are the rope lanes, inverse GPT-J RoPE as vLLM writes it
                p_t = pos[hk.elem_coord(0, 0, 0, 2 * t)].load_scalar(invariant=True)   # int64, low word
                c = hk.iota(ST16)
                k0 = NOPE - DW * (WARPS - 1)
                jc = hk.map(lambda c, p: hk.where(_dl(c) >= k0, p * (D - NOPE) + ((_dl(c) - k0) >> 1), -1),
                            c, p_t)
                ccos = cs.load_elems(jc)
                csin = cs.load_elems(hk.where(jc < 0, -1, jc + (D - NOPE) // 2))
                r = hk.map(lambda x, co, xr, si, c: hk.where(
                    _dl(c) < k0, x, hk.where((c & 1) != 0, x * co - xr * si, x * co + xr * si)),
                    xb, ccos, xb.run_xor(1), csin, c)
                _quant_store(r, mxd, mxs, t, w)
            else:
                _quant_store(xb, mxd, mxs, t, w)

    body.__name__ = name
    return hk.kernel(body, warps=WARPS, grid=lambda p: (p.q.rows // H, 1, 1), name=name, arch=arch)


_KERNELS: Dict[Tuple[float, bool], object] = {}
_DUMMY: Dict[object, tuple] = {}


def kernel_for(scale: float, mxfp8: bool):
    """The kernel for (scale, mxfp8), built once. A scale is a constant of the build, named by its bits."""
    key = (float(scale), bool(mxfp8))
    if key not in _KERNELS:
        import struct  # noqa: PLC0415

        bits = struct.unpack("<Q", struct.pack("<d", float(scale)))[0]
        _KERNELS[key] = kernel(f"sparse_mla_prefill{'_mxfp8' if mxfp8 else ''}_s{bits:016x}",
                               scale=scale, mxfp8=mxfp8)
    return _KERNELS[key]


def supports(q, kv, indices, indptr, attn_sink, out) -> str:
    """'' if the kernel takes these tensors, else why not."""
    import torch  # noqa: PLC0415

    if q.dim() != 3 or tuple(q.shape[1:]) != (H, D):
        return f"q is {tuple(q.shape)}, not [T, {H}, {D}]"
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        return "q, kv and out are bf16"
    if kv.dim() != 2 or kv.shape[1] != D or kv.stride(1) != 1:
        return f"kv is {tuple(kv.shape)}, not [R, {D}] with unit stride along D"
    if not (q.is_contiguous() and out.is_contiguous()):
        return "q and out are contiguous"
    if indices.dtype != torch.int32 or indptr.dtype != torch.int32 or indptr.numel() != q.shape[0] + 1:
        return "indices and indptr are int32, indptr [T + 1]"
    if not (indices.is_contiguous() and indptr.is_contiguous()):
        return "indices and indptr are contiguous"
    if attn_sink.dtype != torch.float32 or attn_sink.numel() != H or not attn_sink.is_contiguous():
        return f"attn_sink is [{H}] fp32, contiguous"
    return ""


def _dummies(dev):
    """The arguments a variant does not read: positions, cos/sin, MXFP8 rows and scales, bf16 rows."""
    import torch  # noqa: PLC0415

    if dev not in _DUMMY:
        _DUMMY[dev] = (torch.zeros(2, dtype=torch.int32, device=dev), torch.zeros(1, D - NOPE, device=dev),
                       torch.zeros(1, D, dtype=torch.float8_e4m3fn, device=dev),
                       torch.zeros(1, D // 32, dtype=torch.int8, device=dev),
                       torch.zeros(1, D, dtype=torch.bfloat16, device=dev))
    return _DUMMY[dev]


def prefill(q, kv, indices, indptr, attn_sink, scale: float, out):
    """out [T, 32, 512] bf16 <- sparse attention of q over the rows of kv each query's CSR list names."""
    import torch  # noqa: PLC0415

    why = supports(q, kv, indices, indptr, attn_sink, out)
    if why:
        raise ValueError(f"hk.ops.sparse_mla.prefill: {why}")
    T = q.shape[0]
    pz, cz, dz, sz, _ = _dummies(q.device)
    kernel_for(scale, False)(q.view(T * H, D), kv, indices, indptr, attn_sink, out.view(T * H, D),
                             pz, cz, dz, sz)
    return out


def prefill_mxfp8(q, kv, indices, indptr, attn_sink, scale: float, positions, cos_sin_cache, mx_data,
                  mx_scale):
    """`prefill`'s rows as wo_a reads them, and no bf16 rows: positions [T] int64, cos_sin_cache [P, 64]
    fp32 (cos | sin), mx_data [T, 32 * 512] e4m3 and mx_scale [T, 512] uint8 E8M0 (row-major), as
    vLLM's rocm_inverse_rope_mxfp8_rows writes them from `prefill`'s output."""
    import torch  # noqa: PLC0415

    T = q.shape[0]
    why = supports(q, kv, indices, indptr, attn_sink, q)
    if not why and (positions.dtype != torch.int64 or positions.numel() != T or not positions.is_contiguous()):
        why = "positions are [T] int64, contiguous"
    if not why and (cos_sin_cache.dtype != torch.float32 or cos_sin_cache.dim() != 2
                    or cos_sin_cache.shape[1] != D - NOPE or not cos_sin_cache.is_contiguous()):
        why = f"cos_sin_cache is [P, {D - NOPE}] fp32, contiguous"
    if not why and (mx_data.dtype != torch.float8_e4m3fn or tuple(mx_data.shape) != (T, H * D)
                    or not mx_data.is_contiguous()):
        why = f"mx_data is [T, {H * D}] float8_e4m3fn, contiguous"
    if not why and (mx_scale.dtype != torch.uint8 or tuple(mx_scale.shape) != (T, H * D // 32)
                    or not mx_scale.is_contiguous()):
        why = f"mx_scale is [T, {H * D // 32}] uint8, contiguous"
    if why:
        raise ValueError(f"hk.ops.sparse_mla.prefill_mxfp8: {why}")
    ob = _dummies(q.device)[4]
    kernel_for(scale, True)(q.view(T * H, D), kv, indices, indptr, attn_sink, ob, positions.view(torch.int32),
                            cos_sin_cache, mx_data.view(T, H * D), mx_scale.view(torch.int8).view(T, H * D // 32))
    return mx_data, mx_scale
