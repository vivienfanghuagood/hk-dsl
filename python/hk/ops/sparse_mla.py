"""DeepSeek-V4.1's sparse MLA prefill attention on gfx950 (CDNA4, MI355X), written in the DSL.

The op vLLM's ROCm path runs for a prefill chunk (`rocm_sparse_attn_prefill`, AITER's opus kernel at
T >= 1024 queries), on the same inputs:

  * q [T, 32, 512] bf16 -- a TP2 rank's 32 heads; the row is 448 nope + 64 rope dims;
  * kv [R, 512] bf16 -- the rows vLLM gathered out of its fp8 caches; V is K, the whole row;
  * indices / indptr int32 -- per query, the rows it attends (CSR): its sliding window and the
    indexer's top-k; indices may run past indptr[T] (vLLM over-allocates);
  * attn_sink [32] fp32 -- exp(sink) joins each head's softmax denominator;
  * out [T, 32, 512] bf16.

`prefill_mxfp8` also writes what the layer hands to wo_a: the rows inverse-RoPE'd on lanes 448-511
(GPT-J pairs) and MXFP8-quantized, one E8M0 scale a 32 lanes -- the arithmetic of vLLM's
`_inverse_rope_mxfp8_quant_kernel` on the bf16 rows, bit for bit.

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

The sink and 1/l at the end; Oᵀ stored transposed. The MXFP8 epilogue re-reads the stored bf16 rows
(each warp a quarter of the heads, all of a head's loads first) as register strips.

Measured on an MI355X against the kernels on vLLM's path, DSV4.1-Flash TP2 calls captured from a
real-weight run (kda/sparse-mla-prefill): top-k layers at T 8192, 640 rows a query, 1.11-1.13x opus
(attention), 1.13x AITER's fused fp8 sparse_mla_fwd and 1.29x opus + vLLM's quant (with the MXFP8
rows); sliding-window layers 1.15-1.20x the faster of opus and AITER (attention). At T <= 256 one
query a workgroup leaves the machine half empty and opus is faster: `MIN_T`.
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

HPW = H // WARPS                                   # heads a warp quantizes in the epilogue
S8 = hk.RS(fp32, 1, D, 8)                          # a head's row, a run of 8 a lane
S8I = hk.RS(i32, 1, D, 8)
S1 = hk.RS(fp32, 1, D - NOPE, 1)                   # the 64 rope lanes, one a lane
S1I = hk.RS(i32, 1, D - NOPE, 1)
#: vLLM's _mxfp8_scale_bits, as written there: the biased E8M0 exponent that puts amax at the top
#: of e4m3's range. (From amax's bits -- hk.mx_quant_store's rule -- differs from it in about one
#: scale in four million: Triton's log2 just past 1.75 x 2^e.)
BITS = ("fminf(fmaxf(ceilf(__builtin_log2f(fmaxf(x0, 1.1754943508222875e-38f) / 448.0f)) + 127.0f, "
        "0.0f), 254.0f)")


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

        g.load(Ks[0], kv.at[0, 0], wait=False, gather=(ind, a))
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
                # rows past the query's (the next query's, or past the indices) are read and masked
                g.load(Ks[1 - s], kv.at[0, 0], wait=False, gather=(ind, a + (blk + 1) * KB))
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
        o[hk.elem_coord(0, 0, t * H, w * DW)].store(hk.transpose(O))
        if mxfp8:
            # the MXFP8 rows from the bf16 rows just stored, as vLLM's pair reads them: each warp a
            # quarter of the heads. Nothing read these rows before (they are this query's), so no
            # vector-L1 line is stale; the stores have landed (vmcnt) before the barrier.
            hk.arch.vm_wait(0)
            hk.arch.barrier()
            p_t = pos[hk.elem_coord(0, 0, 0, 2 * t)].load_scalar(invariant=True)   # int64, low word
            c1 = hk.iota(S1I)
            ccos = cs.load_elems(hk.map_cpp(f"x1 * {D - NOPE} + (x0 >> 1)", c1, p_t, dtype=i32))
            csin = cs.load_elems(hk.map_cpp(f"x1 * {D - NOPE} + {(D - NOPE) // 2} + (x0 >> 1)", c1, p_t,
                                            dtype=i32))
            mate = hk.map_cpp("x0 ^ 1", c1, dtype=i32)
            c8 = hk.iota(S8I)
            # mxd / mxs are [T * 32, 512] / [T * 32, 16]: a row a head of a query, the stores by row
            i8 = hk.map_cpp(f"x0 < {NOPE} ? x0 : -1", c8, dtype=i32)
            s8 = hk.map_cpp(f"(x0 < {NOPE} && (x0 & 31) == 0) ? (x0 >> 5) : -1", c8, dtype=i32)
            i1 = hk.map_cpp(f"x0 + {NOPE}", c1, dtype=i32)
            s1 = hk.map_cpp(f"(x0 & 31) == 0 ? {NOPE // 32} + (x0 >> 5) : -1", c1, dtype=i32)
            # every row's loads first: the compiler does not move row i+1's loads above row i's stores
            x8s = [o.at[t * H + w * HPW + i, 0].load(S8) for i in range(HPW)]
            x1s = [o.at[t * H + w * HPW + i, NOPE].load(S1) for i in range(HPW)]
            for i in range(HPW):
                hrow = t * H + w * HPW + i
                # the nope lanes: groups 0-13 of the run-8 row
                a8 = hk.map_cpp("fabsf(x0)", x8s[i]).group_max(32)
                b8 = hk.map_cpp(BITS, a8)
                q8 = hk.map_cpp("x0 * __builtin_exp2f(127.0f - x1)", x8s[i], b8)
                mxd.store_elems(q8, i8, runs=True, row=hrow)
                mxs.store_elems(b8, s8, row=hrow)
                # the rope lanes, one a lane: inverse GPT-J RoPE as vLLM writes it, then groups 14-15
                y1 = x1s[i].lane_gather(mate)
                r1 = hk.map_cpp("(x4 & 1) ? (x0 * x1 - x2 * x3) : (x0 * x1 + x2 * x3)",
                                x1s[i], ccos, y1, csin, c1, dtype=fp32)
                a1 = hk.map_cpp("fabsf(x0)", r1).group_max(32)
                b1 = hk.map_cpp(BITS, a1)
                q1 = hk.map_cpp("x0 * __builtin_exp2f(127.0f - x1)", r1, b1)
                mxd.store_elems(q1, i1, row=hrow)
                mxs.store_elems(b1, s1, row=hrow)

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


def prefill(q, kv, indices, indptr, attn_sink, scale: float, out):
    """out [T, 32, 512] bf16 <- sparse attention of q over the rows of kv each query's CSR list names."""
    import torch  # noqa: PLC0415

    why = supports(q, kv, indices, indptr, attn_sink, out)
    if why:
        raise ValueError(f"hk.ops.sparse_mla.prefill: {why}")
    T = q.shape[0]
    dev = q.device
    if dev not in _DUMMY:
        _DUMMY[dev] = (torch.zeros(2, dtype=torch.int32, device=dev), torch.zeros(1, D - NOPE, device=dev),
                       torch.zeros(1, D, dtype=torch.float8_e4m3fn, device=dev),
                       torch.zeros(1, D // 32, dtype=torch.int8, device=dev))
    pz, cz, dz, sz = _DUMMY[dev]
    kernel_for(scale, False)(q.view(T * H, D), kv, indices, indptr, attn_sink, out.view(T * H, D),
                             pz, cz, dz, sz)
    return out


def prefill_mxfp8(q, kv, indices, indptr, attn_sink, scale: float, out, positions, cos_sin_cache,
                  mx_data, mx_scale):
    """`prefill`, and the MXFP8 rows wo_a reads: positions [T] int64, cos_sin_cache [P, 64] fp32
    (cos | sin), mx_data [T, 32 * 512] e4m3 and mx_scale [T, 512] uint8 E8M0 (row-major), as
    vLLM's rocm_inverse_rope_mxfp8_rows writes them. `out` holds the bf16 rows."""
    import torch  # noqa: PLC0415

    why = supports(q, kv, indices, indptr, attn_sink, out)
    T = q.shape[0]
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
    kernel_for(scale, True)(q.view(T * H, D), kv, indices, indptr, attn_sink, out.view(T * H, D),
                            positions.view(torch.int32), cos_sin_cache, mx_data.view(T * H, D),
                            mx_scale.view(torch.int8).view(T * H, D // 32))
    return mx_data, mx_scale
