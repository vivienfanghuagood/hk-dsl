"""One decode-attention source for RDNA (gfx1100, gfx1201) and CDNA (gfx950).

hk.ops.paged_decode's walk -- vLLM's TRITON_ATTN cache read in place, the GQA group packed into
one 16-row query tile, the KV axis cut into splits -- written once, in tiles whose shapes are
legal operands on both matrix units (WMMA 16x16x16, wave32; MFMA 16x16x32, wave64), so the same
source builds for both. A workgroup of head_dim / 64 warps, head_dim split between them; per KV block of 32
keys:

    K, V      the block's pages' bands copied global -> LDS by the pair (`g.load`)
    Sᵀ  = K Qᵀ                   [32 keys, 16 queries]: each warp its half of head_dim, in
                                  32-column chunks; the halves added through LDS
    softmax over Sᵀ's columns, online (both warps the same)
    Oᵀ += Vᵀ Pᵀ                  [16-row chunks of the warp's half of head_dim, 16]

Two lines are the target's (`_p_operand`, `_v_operand`): how the scores' accumulator becomes
the PV mma's operand, and how a chunk of V is read as that mma's transposed operand. Why, in
the hardware: an MFMA accumulator's registers already *are* a stride-4 operand fragment
(gfx950 reads Pᵀ in place, `Tile.as_operand`) and MFMA takes Vᵀ from a col-layout LDS read; a
WMMA accumulator is not in an operand's lane layout (the bf16 cast relayouts it) and a col-layout
16-bit LDS read on RDNA is 16 scalar ds_read_u16 a tile (it spilled at head_dim 128), so RDNA
reads V row-wise and transposes in registers. Everything else -- tile shapes, the copies, the
softmax, the split partials, the grid -- is one source.

The split head_dim is what makes one source fit both register files: on RDNA a WMMA operand is
duplicated across the wave's two half-waves, so one warp holding a d128 row's Oᵀ, a K chunk and a
V chunk took 256 VGPRs and spilled; gfx950's wave64 held the same in 78.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .. import dsl as hk
from ..types import bf16, fp32, i32
from .paged_prefill import _kv_at

KV_BLOCK = 32
Q_TILE = 16
NEG_INF = float("-inf")


def _cdna() -> bool:
    from ..target import current_arch  # noqa: PLC0415

    return current_arch().startswith("gfx9")


@hk.func
def _p_operand(s, p, cdna: bool):
    """Sᵀ (an fp32 accumulator) into `p`, the PV mma's bf16 operand -- target-specific line 1."""
    if cdna:
        s.as_operand(bf16, into=p)          # gfx950: the accumulator's registers, read in place
    else:
        s.to(bf16, into=p)                  # RDNA: a cast that moves the data into an operand


@hk.func
def _v_operand(v_chunk, cdna: bool, valid, masked: bool):
    """A [32 keys, 16 d] chunk of V in LDS as the PV mma's Vᵀ -- target-specific line 2. With
    `masked` (a hybrid model's pages: NaN past the sequence's end) its rows from `valid` on are
    zeroed: P = 0 times NaN is NaN."""
    if masked:
        if valid < KV_BLOCK:
            # zeroed in LDS, through a plain row tile: gfx950's operand tile has its rows
            # permuted (stride-4 fragments), where a row fill cannot be applied
            z = v_chunk.load(hk.RT(bf16, KV_BLOCK, 16, "row"), wait=False)
            hk.lds.wait_for(z)
            z.lower_fill_(valid, 0.0)
            v_chunk.store(z)
    v = v_chunk.load(hk.RT(bf16, KV_BLOCK, 16, "col" if cdna else "row"), wait=False)
    hk.lds.wait_for(v)
    if cdna:
        return v.T                          # gfx950: MFMA reads Vᵀ as a col tile's registers
    return hk.transpose_sep(v)              # RDNA: row-wise read, transposed in registers


def walk_kernel(name: str, *, head_dim: int = 128, group: int = 8, page: int = 16,
                kv_order: str = "planes", splits: int = 1, scale: Optional[float] = None):
    """The split walk, head_dim / 64 warps a (split, kv head, request): the tensors of
    hk.ops.paged_decode.walk_kernel -- `qh` (1, tokens, heads, head_dim), contiguous here
    (gfx950's strided global carries a row stride only), the cache `kv` in its `kv_order` view,
    `block_table`, `seq_lens`, `q_start`, `out` (tokens, kv_heads, group, head_dim), and the
    split scratch `o_part` / `ml_part`."""
    D = head_dim
    if D % 64 or D > 256 or page % 16 or not 1 <= group <= Q_TILE:
        raise ValueError(f"head_dim {D}: 64, 128, 192 or 256; page {page}: a multiple of 16; "
                         f"group {group}: 1..{Q_TILE}")
    nw = D // 64                                     # warps: 64 columns of head_dim each
    dh = 64                                          # a warp's share of head_dim
    band = math.gcd(page, KV_BLOCK)
    ppb = KV_BLOCK // band
    masked = page >= KV_BLOCK and page > 16          # hybrid pages: NaN past the end
    scale_l2e = (scale if scale is not None else D ** -0.5) * math.log2(math.e)
    cdna = _cdna()

    T_Q = hk.RT(bf16, Q_TILE, 32, "row")             # a chunk of the group's queries, [q, d]
    T_K = hk.RT(bf16, KV_BLOCK, 32, "row")           # a chunk of a block of K, [kv, d]
    T_S = hk.RT(fp32, KV_BLOCK, Q_TILE, "col")       # Sᵀ, [kv, q]
    T_P = (hk.RT(bf16, KV_BLOCK, Q_TILE, "col", frag=(32, 16, 4)) if cdna
           else hk.RT(bf16, KV_BLOCK, Q_TILE, "col"))  # Pᵀ, the PV operand
    T_O = hk.RT(fp32, 16, Q_TILE, "col")             # a 16-row chunk of Oᵀ, [d, q]
    T_VEC = hk.row_vec(T_S)

    def body(qh: hk.Tensor[bf16], kv: hk.Tensor[bf16], block_table: hk.Tensor[i32],
             seq_lens: hk.Tensor[i32], q_start: hk.Tensor[i32], out: hk.Tensor[bf16],
             o_part: hk.Tensor[bf16], ml_part: hk.Tensor[fp32]):
        split = hk.arch.block_idx.x
        kv_head = hk.arch.block_idx.y
        req = hk.arch.block_idx.z
        tok = q_start[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
        q_len = q_start[hk.elem_coord(0, 0, 0, req + 1)].load_scalar(invariant=True) - tok
        rows = group * hk.min(q_len, 1)
        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
        n_blk = hk.cdiv(seq_len, KV_BLOCK)
        n_pg = hk.max(hk.cdiv(seq_len, page), 1)
        per = hk.max(hk.cdiv(n_blk, splits), 1)
        lo = split * per
        hi = hk.min(n_blk, lo + per)
        k_dep, k_col = _kv_at(kv_order, D, kv_head, False)
        v_dep, v_col = _kv_at(kv_order, D, kv_head, True)
        half = hk.arch.warp_id()                     # this warp's 64 columns of head_dim

        g = hk.group(nw)
        q_s = hk.lds.alloc(hk.ST(bf16, Q_TILE, D), name="q_s")
        k_s = hk.lds.alloc(hk.ST(bf16, KV_BLOCK, D), name="k_s")
        v_s = hk.lds.alloc(hk.ST(bf16, KV_BLOCK, D), name="v_s")
        xs = hk.lds.alloc(hk.ST(fp32, KV_BLOCK, Q_TILE, max(nw, 2)), name="xs")  # partial Sᵀ
        k_b = [k_s.subtile(band, D, p, 0) for p in range(ppb)]
        v_b = [v_s.subtile(band, D, p, 0) for p in range(ppb)]

        g.load(q_s, qh[hk.elem_coord(0, tok, kv_head * group, 0)])
        hk.arch.barrier()

        def copy(b):
            # block b's bands (clamped into the request's pages: past the end it rereads the last)
            for p in range(ppb):
                i = (b * (KV_BLOCK // page) + p if page <= KV_BLOCK
                     else (b * KV_BLOCK + p * band) // page)
                pg = block_table[hk.elem_coord(0, 0, req, hk.min(i, n_pg - 1))].load_scalar(
                    invariant=True)
                r0 = 0 if page <= KV_BLOCK else (b * KV_BLOCK + p * band) % page
                g.load(k_b[p], kv[hk.elem_coord(pg, k_dep, r0, k_col)])
                g.load(v_b[p], kv[hk.elem_coord(pg, v_dep, r0, v_col)])

        O = [hk.zeros(T_O) for _ in range(dh // 16)]
        m = hk.neg_infty(T_VEC)
        m_old = hk.neg_infty(T_VEC)
        ell = hk.zeros(T_VEC)
        S = hk.zeros(T_S)
        P = hk.zeros(T_P)

        for bi in range(hk.max(hi - lo, 0)):
            b = lo + bi
            hk.arch.barrier()                      # the last block's LDS reads are done
            copy(b)
            hk.arch.barrier()
            S.zero_()
            for c in range(dh // 32):
                kr = k_s.subtile(KV_BLOCK, 32, 0, half * (dh // 32) + c).load(T_K, wait=False)
                qr = q_s.subtile(Q_TILE, 32, 0, half * (dh // 32) + c).load(T_Q, wait=False)
                hk.lds.wait_for(kr, qr)
                S += kr @ qr.T
            xs[half].store(S)                      # the warps' partial Sᵀ, added
            hk.arch.barrier(drain=True)
            for w in range(1, nw):
                S += xs[(half + w) % nw].load(T_S)
            S *= scale_l2e
            valid = seq_len - b * KV_BLOCK
            if valid < KV_BLOCK:
                S.lower_fill_(valid, NEG_INF)
            m.copy(into=m_old)
            m.max_(S.max(axis=0))
            S -= m
            S.exp2_()
            alpha = m_old - m
            alpha.exp2_()
            ell *= alpha
            for o in O:
                o *= hk.as_vec(alpha, T_O)
            ell += S.sum(axis=0)
            _p_operand(S, P, cdna)
            for c in range(dh // 16):
                O[c] += _v_operand(v_s.subtile(KV_BLOCK, 16, 0, half * (dh // 16) + c), cdna,
                                   valid, masked) @ P

        inv = hk.full(T_VEC, 1.0)
        ell_s = ell.copy()
        ell_s += 1e-30
        inv /= ell_s
        for c in range(dh // 16):
            O[c] *= hk.as_vec(inv, T_O)
        col0 = half * dh
        if splits == 1:
            for c in range(dh // 16):
                out[hk.elem_coord(tok, kv_head, 0, col0 + 16 * c)].store(hk.transpose(O[c]),
                                                                         bounded=rows)
            return
        row = req * splits + split
        for c in range(dh // 16):
            o_part[hk.elem_coord(row, kv_head, 0, col0 + 16 * c)].store(hk.transpose(O[c]),
                                                                        bounded=True)
        if half == 0:
            ml_part[hk.elem_coord(row, kv_head, 0, 0)] = m
            ml_part[hk.elem_coord(row, kv_head, 1, 0)] = ell

    body.__name__ = name
    return hk.kernel(body, warps=nw, name=name,
                     grid=lambda p: (splits, p.out.depth, p.seq_lens.cols))


_WALK: Dict[tuple, object] = {}


def walk_for(head_dim: int, group: int, page: int, kv_order: str, splits: int,
             scale: Optional[float] = None):
    """The walk for one shape on the current target, built on first use."""
    from ..target import current_arch  # noqa: PLC0415

    key = (current_arch(), head_dim, group, page, kv_order, splits, scale)
    if key not in _WALK:
        tag = "" if scale is None else "_s%08x" % (hash(scale) & 0xffffffff)
        _WALK[key] = walk_kernel(
            f"paged_decode_port_d{head_dim}_g{group}_p{page}_{kv_order}_s{splits}{tag}",
            head_dim=head_dim, group=group, page=page, kv_order=kv_order, splits=splits,
            scale=scale)
    return _WALK[key]
