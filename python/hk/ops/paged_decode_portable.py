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

The source is the same for both targets. The two places the hardware differs are the DSL's:
`S.as_operand(bf16)` is the scores' accumulator read in place as the PV mma's operand on gfx950
and its operand cast on RDNA (#212), and a col-layout read of V from LDS (`load(RT(..., "col"))`,
then `.T`) is a stride-4 operand read on gfx950 and a row read transposed in registers on RDNA
(#213). One capability is gfx950's alone today: its Vᵀ operand comes from whole 32-row LDS
subtiles, so a 32-key block of V must arrive in one copy -- gfx950 takes pages of whole 32-key
blocks (32, 64, 544), not 16, 784 or 2064 (#218).

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
#: Columns of V (rows of Oᵀ) a PV mma takes: 16, the stride-4 operand's own width on gfx950, and
#: half the registers of 32 on RDNA, where the masked pages' clamp of a 32-wide chunk spilled
VC = 16
NEG_INF = float("-inf")


def _cdna() -> bool:
    from ..target import current_arch  # noqa: PLC0415

    return current_arch().startswith("gfx9")


@hk.func
def _v_operand(v_chunk, masked: bool, valid):
    """A [32 keys, VC d] chunk of V in LDS as the PV mma's Vᵀ: a col-layout read, `.T`. gfx950
    reads it as stride-4 operand fragments; RDNA as row-major b128s transposed in registers
    (#213) -- one line for both. With `masked` (a hybrid model's pages: NaN past the sequence's
    end, where P is 0, and 0 times NaN is NaN) the sequence's last block (`valid` < 32 keys) is
    clamped to bf16's finite range: a NaN becomes finite, a value does not change. Elementwise,
    so it holds in any operand layout; only the last block, because only it has rows past the
    end."""
    v = v_chunk.load(hk.RT(bf16, KV_BLOCK, VC, "col"))
    if masked:
        if valid < KV_BLOCK:
            v.maximum(-3.0e38, into=v)
            v.minimum(3.0e38, into=v)
    return v.T


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
    cdna = _cdna()
    dh = 64                                          # a warp's share of head_dim
    band = math.gcd(page, KV_BLOCK)                   # rows a copy brings from one page
    if not cdna:
        # RDNA copies a block in 16-row bands whatever the page: a 32-row band's staging took the
        # 8 VGPRs over 256 that spilled at head_dim 256 on gfx1201 (clang 23, a GQA group of 6)
        band = min(band, 16)
    ppb = KV_BLOCK // band
    masked = page >= KV_BLOCK and page > 16          # hybrid pages: NaN past the end
    scale_l2e = (scale if scale is not None else D ** -0.5) * math.log2(math.e)
    if cdna and band < KV_BLOCK:
        # gfx950's Vᵀ operand is read from an LDS tile laid out in 32x32 subtiles (32x16
        # stride-4 fragments): a block's V has to arrive as whole 32-row subtiles, and a page
        # of 16 (or a page that is not whole blocks, 784, 2064) brings it in 16-row bands
        raise ValueError(f"page={page}: on gfx950 a page of a whole number of 32-key blocks "
                         f"(a block of V is one 32-row LDS subtile there)")

    T_Q = hk.RT(bf16, Q_TILE, 32, "row")             # a chunk of the group's queries, [q, d]
    T_K = hk.RT(bf16, KV_BLOCK, 32, "row")           # a chunk of a block of K, [kv, d]
    T_S = hk.RT(fp32, KV_BLOCK, Q_TILE, "col")       # Sᵀ, [kv, q]
    T_O = hk.RT(fp32, VC, Q_TILE, "col")             # a VC-row chunk of Oᵀ, [d, q]
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
                i = (b * KV_BLOCK + p * band) // page
                pg = block_table[hk.elem_coord(0, 0, req, hk.min(i, n_pg - 1))].load_scalar(
                    invariant=True)
                r0 = 0 if page == band else (b * KV_BLOCK + p * band) % page
                g.load(k_b[p], kv[hk.elem_coord(pg, k_dep, r0, k_col)])
                g.load(v_b[p], kv[hk.elem_coord(pg, v_dep, r0, v_col)])

        O = [hk.zeros(T_O) for _ in range(dh // VC)]
        m = hk.neg_infty(T_VEC)
        m_old = hk.neg_infty(T_VEC)
        ell = hk.zeros(T_VEC)
        S = hk.zeros(T_S)
        P = S.as_operand(bf16)                     # Pᵀ, the PV operand: its type is the target's

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
            S.as_operand(bf16, into=P)
            for c in range(dh // VC):
                O[c] += _v_operand(v_s.subtile(KV_BLOCK, VC, 0, half * (dh // VC) + c),
                                   masked, valid) @ P

        inv = hk.full(T_VEC, 1.0)
        ell_s = ell.copy()
        ell_s += 1e-30
        inv /= ell_s
        for c in range(dh // VC):
            O[c] *= hk.as_vec(inv, T_O)
        col0 = half * dh
        if splits == 1:
            for c in range(dh // VC):
                out[hk.elem_coord(tok, kv_head, 0, col0 + VC * c)].store(hk.transpose(O[c]),
                                                                         bounded=rows)
            return
        row = req * splits + split
        for c in range(dh // VC):
            o_part[hk.elem_coord(row, kv_head, 0, col0 + VC * c)].store(hk.transpose(O[c]),
                                                                        bounded=True)
        if half == 0:
            ml_part[hk.elem_coord(row, kv_head, 0, 0)] = m
            ml_part[hk.elem_coord(row, kv_head, 1, 0)] = ell

    body.__name__ = name
    return hk.kernel(body, warps=nw, name=name,
                     grid=lambda p: (splits, p.out.depth, p.seq_lens.cols))


#: The walk's splits for a batch: about TARGET_WORKGROUPS (request, kv head, split) workgroups,
#: a power of two from MIN_SPLITS to MAX_SPLITS, from the batch alone (a CUDA graph's replays see
#: other lengths). Its own plan, not hk.ops.paged_decode's: a split of this walk does less work
#: than one of the RDNA-tuned walk, so it wants more of them (W7900D, MiniCPM5 B4 @16k: 8 splits
#: 197 us, 16 splits 128 us, vLLM's Triton 124 us).
TARGET_WORKGROUPS = 256
MIN_SPLITS = 16
MAX_SPLITS = 64


def plan_splits(n_reqs: int, n_kv_heads: int, max_blocks: Optional[int] = None) -> int:
    """Splits for a batch of `n_reqs` requests over `n_kv_heads` kv heads (see TARGET_WORKGROUPS),
    capped, when known, by the longest request's KV blocks."""
    have = max(1, n_reqs * n_kv_heads)
    want = 1 << max(0, (TARGET_WORKGROUPS // have).bit_length() - 1)
    want = min(max(want, MIN_SPLITS), MAX_SPLITS)
    if max_blocks is not None:
        want = max(1, min(want, max_blocks))
    return want


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
