"""Paged attention on gfx950 (CDNA4, MI355X): the split-KV decode.

The same contract as hk.ops.paged's kernels -- the cache is
`(num_blocks, num_kv_heads, page, head_dim)`, each of K and V, a page of one
head a contiguous (or row-strided) `(page, head_dim)` tile; `block_table`
and `seq_lens` are int32 -- on gfx950's matrix core:

  * **The scores are kept transposed.** Sᵀ = K·Qᵀ is a (page x 16) fp32
    accumulator in 16x16 fragments (the 16x16x32 MFMA), its 16 columns the
    GQA group's query rows padded to a tile. The softmax's statistics are one
    value per query, a row vector of Sᵀ, and Oᵀ (head_dim x 16) takes them as
    one factor a column.
  * **Pᵀ is read in place** (`Tile.as_operand`): two 16-row accumulators are
    one 32x16 operand fragment with a lane run of 4, and V is read out of LDS
    with the same run by the transposing ds_read_b64_tr_b16, so P·V needs no
    register moves.
  * **K and V come by DMA.** Each page is copied into an LDS ring
    (`g.load(wait=False)`) a page ahead of the math, its vmcnt derived; a
    copy is bounded to the sequence's rows, so a page's tail -- whatever the
    allocator left there -- reads zero, and a page past the split's end
    copies nothing. The page table and the lengths are read through the
    scalar cache (`load_scalar(invariant=True)`): a vector load's wait sits
    in vmcnt's queue behind every copy issued before it, and held the ring
    to one page in flight.

One warp a workgroup, a workgroup a (split, kv head, request): decode is a
read of the cache, and the machine is filled by splitting the KV axis
(hk.ops.paged.plan_splits), not by warps sharing a page. The splits merge
in hk.ops.paged's merge kernel, which builds for gfx950 as it is.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .. import dsl as hk
from ..types import bf16, fp32, i32

NEG_INF = float("-inf")

#: The cache's page: hk.ops.paged.KV_BLOCK.
PAGE = 32
#: Query rows of a tile: the GQA group, padded.
Q_TILE = 16
#: Head dims with a schedule.
HEAD_DIMS = (64, 128)
#: Pages a split takes at least, so that a short sequence under a split
#: count planned for a long one walks a few splits of four pages, and the
#: merge reads only those: a one-request decode of 512 to 8192 tokens is
#: then within 2-5% of its best split count at every length, where one
#: split count for all was up to 40% off (docs/history, the gfx950 paged
#: chapter).
MIN_SPLIT_PAGES = 4
#: One-warp workgroups gfx950's decode plan aims for (hk.ops.paged.plan_splits
#: on gfx950): the best split count measured at every batch from 1 to 128
#: requests and 512 to 8192 tokens put about 256 on the card, one a CU, for
#: contexts of 4096 and more; MIN_SPLIT_PAGES takes the shorter ones down
#: to fewer splits on the device (tools/hk-bench/paged_cdna.py).
TARGET_WORKGROUPS = 256

#: The pages' LDS layouts: K's 16x32 operand fragments read out of 16x64
#: subtiles, V's 32x16 stride-4 ones out of 32x32, so a page's copy reads
#: each row in 128- and 64-byte runs, where the fragments' own layouts made
#: them 64 and 32 -- every L2 line asked for twice, and a non-temporal copy
#: fetching it twice. Both, with nontemporal copies: 6.15 to 6.9 TB/s at
#: 128 x 8192 (aiter's gluon kernel: 6.6).
K_SHAPE, V_SHAPE = "16x64", "32x32"
#: What `ablate` can take out, for timing: "math" leaves the copies and the
#: LDS reads and nothing after them (wrong answers on purpose).
ABLATIONS = ("math",)


def decode_kernel(name: str, *, head_dim: int = 128, splits: int = 1,
                  mode: str = "fused", stages: int = 2, ablate: tuple = (),
                  scale: Optional[float] = None, arch: Optional[str] = None):
    """The paged decode for one (split, kv head, request) a workgroup.

    `mode` is the output contract, one per hk.ops.paged kernel this stands in
    for on gfx950:

      "plain"  (q, k, v, table, lens, o): q and o (requests, kv heads, rows,
               head_dim), one split, the answer (paged.KERNELS).
      "split"  (q, k, v, table, lens, o_part, ml_part): each split's
               unnormalised fp32 Oᵀ, transposed, and its (m, l), for
               paged.merge_splits (paged.split_kernel).
      "fused"  (q, k, v, table, lens, out, o_part, ml_part): the engine's
               (tokens, kv heads, group, head_dim) query and output; one
               split writes the answer, more write each split's normalised
               bf16 partial and (m, l) for paged.merge_kernel
               (paged.fused_kernel).

    `stages` LDS buffers each of K and V: page t+stages-1 is copied while
    page t is worked. Two is the measured best: a deeper ring costs a
    workgroup's slot on the CU and the walk is not short of pages in flight
    (docs/history, the gfx950 paged chapter).
    """
    D = head_dim
    if D not in HEAD_DIMS:
        raise ValueError(f"head_dim={D}: have {HEAD_DIMS}")
    if mode not in ("plain", "split", "fused"):
        raise ValueError(f"mode {mode!r}")
    if stages < 2:
        raise ValueError(f"stages={stages}: two buffers or more")
    bad = set(ablate) - set(ABLATIONS)
    if bad:
        raise ValueError(f"unknown ablation {sorted(bad)}")
    scale_l2e = (scale if scale is not None else D ** -0.5) * math.log2(math.e)
    n_split = 1 if mode == "plain" else splits

    T_Q = hk.RT(bf16, Q_TILE, D, "row")        # the group's queries, [q, d]
    T_K = hk.RT(bf16, PAGE, D, "row")          # a page of K, [kv, d]
    T_V = hk.RT(bf16, PAGE, D, "col")          # a page of V, [kv, d]
    T_S = hk.RT(fp32, PAGE, Q_TILE, "col")     # Sᵀ, [kv, q]
    T_P = hk.RT(bf16, PAGE, Q_TILE, "col", frag=(32, 16, 4))   # Pᵀ, the operand
    T_O = hk.RT(fp32, D, Q_TILE, "col")        # Oᵀ, [d, q]
    T_VEC = hk.row_vec(T_S)                    # one value per query
    ST = hk.ST(bf16, PAGE, D)

    @hk.func
    def walk(q, k_cache, v_cache, block_table, seq_lens, finish):
        g = hk.group(1)
        Ks = hk.lds.alloc(ST, count=stages, name="Ks", shape=K_SHAPE)
        Vs = hk.lds.alloc(ST, count=stages, name="Vs", shape=V_SHAPE)
        split = hk.arch.block_idx.x
        kv_head = hk.arch.block_idx.y
        req = hk.arch.block_idx.z

        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
        n_pages = hk.cdiv(seq_len, PAGE)
        per = hk.max(hk.cdiv(n_pages, n_split), MIN_SPLIT_PAGES)
        lo = split * per
        n = hk.max(hk.min(n_pages, lo + per) - lo, 0)
        last = hk.max(n_pages - 1, 0)

        def rows_of(t):
            """Rows of the split's page t the sequence has: 0 past the
            split's end, so its copy moves nothing."""
            live = hk.min(hk.max(n - t, 0), 1)
            return live * hk.min(hk.max(seq_len - (lo + t) * PAGE, 0), PAGE)

        def page_of(t):
            return block_table[hk.elem_coord(
                0, 0, req, hk.min(lo + t, last))].load_scalar(invariant=True)

        def copy(t, buf, page):
            rows = rows_of(t)
            g.load(Ks[buf], k_cache[hk.elem_coord(page, kv_head, 0, 0)],
                   wait=False, rows=rows, nontemporal=True)
            g.load(Vs[buf], v_cache[hk.elem_coord(page, kv_head, 0, 0)],
                   wait=False, rows=rows, nontemporal=True)

        qr = q[hk.elem_coord(req, kv_head, 0, 0)].load(T_Q, bounded=True)
        kr = hk.zeros(T_K)
        vr = hk.zeros(T_V)
        S = hk.zeros(T_S)
        P = hk.zeros(T_P)
        O = hk.zeros(T_O)
        m = hk.neg_infty(T_VEC)
        m_old = hk.neg_infty(T_VEC)
        ell = hk.zeros(T_VEC)

        def math_page(buf, t):
            Ks[buf].load(T_K, wait=False, into=kr)
            Vs[buf].load(T_V, wait=False, into=vr)
            hk.lds.wait_for(kr)
            if "math" in ablate:
                hk.lds.wait_for(vr)
                return
            S.zero_()
            S.__iadd__(kr @ qr.T)
            S.__imul__(scale_l2e)
            rows = rows_of(t)
            if rows < PAGE:
                S.lower_fill_(rows, NEG_INF)
            m.copy(into=m_old)
            m.max_(S.max(axis=0))
            S.__isub__(m)
            S.exp2_()
            if m.any_ne(m_old):
                alpha = m_old - m
                alpha.exp2_()
                ell.__imul__(alpha)
                O.__imul__(hk.as_vec(alpha, T_O))
            ell.__iadd__(S.sum(axis=0))
            S.as_operand(bf16, into=P)
            hk.lds.wait_for(vr)
            O.__iadd__(vr.T @ P)

        # The page index a copy needs is read a page ahead of it, so the
        # copy does not wait out a load of the table.
        for j in range(stages - 1):
            copy(j, j, page_of(j))
        nxt = page_of(stages - 1)
        for it in range(hk.cdiv(n, stages)):
            for j in range(stages):
                t = it * stages + j
                # Page t-1's buffer: its reads retired in the last math, and
                # a barrier after them before the copy overwrites it.
                hk.arch.barrier()
                page = nxt
                nxt = page_of(t + stages)
                copy(t + stages - 1, (j + stages - 1) % stages, page)
                g.wait_for(Ks[j], Vs[j])
                hk.arch.barrier()
                math_page(j, t)
        # The ring's last copies are of pages past the split: nothing, but
        # in flight into LDS.
        hk.arch.vm_wait(0)
        finish(O, m, ell, split, kv_head, req)

    def normalise(O, ell):
        # One division a query, then a multiply an element; the floor makes
        # an empty split's (and a padded request's) answer 0, not 0/0.
        inv = hk.full(T_VEC, 1.0)
        ell.__iadd__(1e-30)
        inv.__itruediv__(ell)
        O.__imul__(hk.as_vec(inv, T_O))

    if mode == "plain":
        def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16],
                 v_cache: hk.Tensor[bf16], block_table: hk.Tensor[i32],
                 seq_lens: hk.Tensor[i32], o: hk.Tensor[bf16]):
            def finish(O, m, ell, split, kv_head, req):
                normalise(O, ell)
                o[hk.elem_coord(req, kv_head, 0, 0)].store(hk.transpose(O), bounded=True)
            walk(q, k_cache, v_cache, block_table, seq_lens, finish)
    elif mode == "split":
        def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16],
                 v_cache: hk.Tensor[bf16], block_table: hk.Tensor[i32],
                 seq_lens: hk.Tensor[i32], o_part: hk.Tensor[fp32],
                 ml_part: hk.Tensor[fp32]):
            def finish(O, m, ell, split, kv_head, req):
                row = req * n_split + split
                o_part[hk.elem_coord(row, kv_head, 0, 0)].store(hk.transpose(O), bounded=True)
                ml_part[hk.elem_coord(row, kv_head, 0, 0)] = m
                ml_part[hk.elem_coord(row, kv_head, 1, 0)] = ell
            walk(q, k_cache, v_cache, block_table, seq_lens, finish)
    else:
        def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16],
                 v_cache: hk.Tensor[bf16], block_table: hk.Tensor[i32],
                 seq_lens: hk.Tensor[i32], out: hk.Tensor[bf16],
                 o_part: hk.Tensor[bf16], ml_part: hk.Tensor[fp32]):
            def finish(O, m, ell, split, kv_head, req):
                if n_split == 1:
                    normalise(O, ell)
                    out[hk.elem_coord(req, kv_head, 0, 0)].store(
                        hk.transpose(O), bounded=True)
                    return
                # Normalised, its values are a convex mix of V's, so bf16
                # costs the partial what bf16 costs the output; an empty
                # split has l = 0 and O = 0, and the merge weighs it zero.
                row = req * n_split + split
                ml_part[hk.elem_coord(row, kv_head, 0, 0)] = m
                ml_part[hk.elem_coord(row, kv_head, 1, 0)] = ell
                normalise(O, ell)
                o_part[hk.elem_coord(row, kv_head, 0, 0)].store(
                    hk.transpose(O), bounded=True)
            walk(q, k_cache, v_cache, block_table, seq_lens, finish)

    body.__name__ = name
    return hk.kernel(body, arch=arch, warps=1, name=name,
                     grid=lambda p: (n_split, p.q.depth, p.q.batch))


#: Splits the gfx950 merge reads a round.
MERGE_CHUNK = 8


def merge_kernel(name: str, *, head_dim: int = 128, splits: int = 8,
                 arch: Optional[str] = None):
    """The fused decode's merge on gfx950, (seq_lens, out, o_part, ml_part)
    as hk.ops.paged.merge_kernel's: one warp a (request, kv head, query row),
    the row a strip of head_dim, the splits that saw pages read MERGE_CHUNK
    at a time and folded online in registers.

    hk.ops.paged's merge gives a warp 16 columns of a 16-row tile -- of
    which the group's four rows are real -- and on gfx950 it took 3.7 us at
    8 splits and 7.5 at 32, against 3.0-4.0 us for vLLM's and aiter's
    reductions: most of a one-request decode. Here 2.2-2.3 us for up to 8
    splits that saw pages, 3.1 for 16 and 4.7 for 32. A split that saw
    nothing wrote m = -inf and l = 0 and weighs zero; the max is floored so
    that a request with no context at all, a CUDA graph's padding, comes out
    0 rather than 0/0."""
    D = head_dim
    run = D // 64
    if D % 64 or run not in (1, 2):
        raise ValueError(f"head_dim={D}: a strip of 64 or 128")
    T = hk.RS(fp32, 1, D, run=run)
    CV = hk.col_vec(T)

    chunk = min(MERGE_CHUNK, splits)
    if splits % chunk:
        raise ValueError(f"splits={splits}: a whole number of {chunk}-split rounds")

    def body(seq_lens: hk.Tensor[i32], out: hk.Tensor[bf16],
             o_part: hk.Tensor[bf16], ml_part: hk.Tensor[fp32]):
        g = hk.arch.block_idx.x
        kv_head = hk.arch.block_idx.y
        req = hk.arch.block_idx.z
        # The splits that saw pages are the walk's first `active`: a round
        # past them would be all empty splits, weightless, and is not run.
        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
        n_pages = hk.cdiv(seq_len, PAGE)
        active = hk.cdiv(n_pages, hk.max(hk.cdiv(n_pages, splits), MIN_SPLIT_PAGES))
        ms = [hk.zeros(CV) for _ in range(chunk)]
        ls = [hk.zeros(CV) for _ in range(chunk)]
        os_ = [hk.zeros(T) for _ in range(chunk)]
        top = hk.full(CV, -1e30)
        old = hk.zeros(CV)
        acc = hk.zeros(T)
        tot = hk.zeros(CV)
        # A round of `chunk` splits a trip, its reads issued together and
        # folded online. Every split's addresses held at once spilled 110
        # SGPRs at 32 splits, and the next round's reads issued under this
        # one's fold 29. A padded request (no context) folds one round of
        # empty splits and stores 0.
        for k in range(hk.max(hk.cdiv(active, chunk), 1)):
            for j in range(chunk):
                r = req * splits + k * chunk + j
                ml_part[hk.elem_coord(r, kv_head, 0, g)].load(CV, into=ms[j])
                ml_part[hk.elem_coord(r, kv_head, 1, g)].load(CV, into=ls[j])
                o_part[hk.elem_coord(r, kv_head, g, 0)].load(T, into=os_[j])
            top.copy(into=old)
            for m_s in ms:
                top.max_(m_s)
            old.__isub__(top)
            old.exp2_()
            tot.__imul__(old)
            acc.__imul__(old[:, None])
            for m_s, l_s, o_s in zip(ms, ls, os_):
                m_s.__isub__(top)
                m_s.exp2_()
                m_s.__imul__(l_s)
                tot.__iadd__(m_s)
                o_s.__imul__(m_s[:, None])
                acc.__iadd__(o_s)
        tot.__iadd__(1e-30)
        acc.__itruediv__(tot[:, None])
        out[hk.elem_coord(req, kv_head, g, 0)].store(acc)

    body.__name__ = name
    return hk.kernel(body, arch=arch, warps=1, name=name,
                     grid=lambda p: (p.out.rows, p.out.depth, p.out.batch))


_KERNELS: Dict[Any, Any] = {}


def merge_for(head_dim: int, splits: int):
    """The gfx950 merge for (head_dim, splits), built once."""
    key = ("merge", head_dim, splits)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = merge_kernel(f"paged_merge_d{head_dim}_s{splits}_cdna",
                                         head_dim=head_dim, splits=splits)
    return k


def kernel_for(head_dim: int, splits: int = 1, mode: str = "fused", **opts):
    """The gfx950 decode for (head_dim, splits, mode), built once."""
    opts = {k: (tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in opts.items()}
    key = (head_dim, splits, mode, tuple(sorted(opts.items())))
    k = _KERNELS.get(key)
    if k is None:
        tag = {"plain": "", "split": f"_split_s{splits}",
               "fused": f"_fused_s{splits}"}[mode]
        name = f"paged_decode_d{head_dim}{tag}_cdna"
        for kk, vv in sorted(opts.items()):
            val = int(vv) if isinstance(vv, bool) else (
                "".join(f"{x}" for x in vv) if isinstance(vv, tuple) else vv)
            name += "_" + "".join(ch if ch.isalnum() else "p" for ch in f"{kk}{val}")
        k = _KERNELS[key] = decode_kernel(name, head_dim=head_dim,
                                          splits=splits, mode=mode, **opts)
    return k


__all__ = ["decode_kernel", "kernel_for", "PAGE", "Q_TILE", "HEAD_DIMS", "ABLATIONS"]
