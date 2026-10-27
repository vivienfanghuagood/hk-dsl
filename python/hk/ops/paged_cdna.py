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
in this module's merge_kernel, which hk.ops.paged.merge_kernel routes to.
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
#: on gfx950, `target_workgroups`): about 256, one a CU, below 64 (request,
#: kv head) pairs and from 256, and 512 between. Measured with the merge's warps
#: (`merge_warps`) at 1-64 requests, 1-8 kv heads and 1k-128k tokens, one
#: split count a (requests, kv heads) for every length, as a captured step
#: needs; MIN_SPLIT_PAGES takes the shorter sequences down to fewer splits on
#: the device. Before the merge had several warps the plan stopped at 32
#: splits, where one request on 1 kv head and 128k tokens walked 32
#: workgroups for 83 us; 256 splits take 24 (docs/history, the gfx950 paged
#: chapter; tools/hk-bench/paged_cdna.py).
TARGET_WORKGROUPS = 256
TARGET_WORKGROUPS_FULL = 512
#: The most splits gfx950's plan asks for (hk.ops.paged.MAX_SPLITS is
#: gfx1100's).
MAX_SPLITS = 512


def target_workgroups(pairs: int) -> int:
    """The one-warp workgroups to aim for with `pairs` (request, kv head)
    pairs in the batch: 512 from 64 pairs to 255, 256 otherwise. From 256
    pairs on, two splits were 5% faster at 128k tokens and 5% slower at 1k
    than one, and a captured step cannot tell which it will see."""
    return TARGET_WORKGROUPS_FULL if 64 <= pairs < 256 else TARGET_WORKGROUPS

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
                  scale: Optional[float] = None, arch: Optional[str] = None,
                  min_pages: int = MIN_SPLIT_PAGES):
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
        raise ValueError(f"head_dim={D}: gfx950's paged decode has a schedule for "
                         f"{HEAD_DIMS} only (gfx1100's takes multiples of 16)")
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
        per = hk.max(hk.cdiv(n_pages, n_split), min_pages)
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
        if mode == "fused" and n_split > MERGE_CHUNK:
            # The merge reads the rounds of MERGE_CHUNK splits that hold the
            # first `active` (one round for a padded request); a split past
            # them has nothing to write.
            active = hk.cdiv(n_pages, per)
            read = hk.max(hk.cdiv(active, MERGE_CHUNK), 1) * MERGE_CHUNK
            if split < read:
                finish(O, m, ell, split, kv_head, req)
        else:
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
#: The most warps a merge workgroup folds with.
MERGE_WARPS = 16


def merge_warps(splits: int) -> int:
    """Warps of the merge for `splits` splits: about a round of MERGE_CHUNK
    each, so the fold is a round or two long whatever the split count."""
    return min(MERGE_WARPS, max(1, splits // MERGE_CHUNK))


def merge_kernel(name: str, *, head_dim: int = 128, splits: int = 8,
                 arch: Optional[str] = None, min_pages: int = MIN_SPLIT_PAGES,
                 warps: int = 1):
    """The fused decode's merge on gfx950, (seq_lens, out, o_part, ml_part)
    as hk.ops.paged.merge_kernel's: a workgroup a (request, kv head, query
    row), the row a strip of head_dim, the splits that saw pages read
    MERGE_CHUNK at a time and folded online in registers. With `warps`
    warps, warp w folds rounds w, w + warps, ...; the warps' maxima and
    sums are combined through LDS (hk.collective.cross_warp), their strips
    element by element, and warp 0 stores. A single warp folding 256 splits
    took longer than the walk that wrote them.

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
        raise ValueError(f"splits={splits}: a whole number of {chunk}-split rounds -- "
                         f"gfx950's merge reads {chunk} splits at a time (gfx1100's "
                         f"takes any count); round the plan to a power of two")

    def body(seq_lens: hk.Tensor[i32], out: hk.Tensor[bf16],
             o_part: hk.Tensor[bf16], ml_part: hk.Tensor[fp32]):
        g = hk.arch.block_idx.x
        kv_head = hk.arch.block_idx.y
        req = hk.arch.block_idx.z
        # The splits that saw pages are the walk's first `active`: a round
        # past them would be all empty splits, weightless, and is not run.
        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
        n_pages = hk.cdiv(seq_len, PAGE)
        active = hk.cdiv(n_pages, hk.max(hk.cdiv(n_pages, splits), min_pages))
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
        rounds = hk.max(hk.cdiv(active, chunk), 1)
        wid = hk.arch.warp_id() if warps > 1 else 0
        # With several warps, warp w folds rounds w, w + warps, ...
        for kk in range(hk.cdiv(hk.max(rounds - wid, 0), warps) if warps > 1 else rounds):
            k = kk * warps + wid if warps > 1 else kk
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
        if warps > 1:
            sv_max = hk.lds.alloc_vec(hk.lds.shared_vec(CV, warps), name="mmax")
            sv_sum = hk.lds.alloc_vec(hk.lds.shared_vec(CV, warps), name="msum")
            parts = hk.lds.alloc(hk.ST(fp32, max(16, warps), D), name="mparts")
            # One round (a short sequence) is warp 0's alone: nothing to
            # combine. Otherwise the warps' folds, rescaled to the largest
            # max and summed: a warp with no round has max -1e30, total 0
            # and weighs nothing; the strips through LDS element by element,
            # a row a warp, summed by warp 0 in warp order.
            if hk.uniform(rounds > 1):
                big = hk.collective.cross_warp(top, "max", warps, scratch=sv_max)
                top.__isub__(big)
                top.exp2_()
                tot.__imul__(top)
                acc.__imul__(top[:, None])
                hk.collective.cross_warp(tot, "add", warps, scratch=sv_sum).copy(into=tot)
                col = hk.iota(T)
                parts.store_elems(acc, hk.map_cpp(f"x1 * {D} + x0", col, hk.arch.warp_id()))
                hk.arch.barrier(drain=True)
                for w in range(1, warps):
                    acc.__iadd__(parts.load_elems(hk.map_cpp(f"{w * D} + x0", col)))
        tot.__iadd__(1e-30)
        acc.__itruediv__(tot[:, None])
        if warps > 1:
            if hk.arch.warp_id() == 0:
                out[hk.elem_coord(req, kv_head, g, 0)].store(acc)
        else:
            out[hk.elem_coord(req, kv_head, g, 0)].store(acc)

    body.__name__ = name
    return hk.kernel(body, arch=arch, warps=warps, name=name,
                     grid=lambda p: (p.out.rows, p.out.depth, p.out.batch))


_KERNELS: Dict[Any, Any] = {}


def merge_for(head_dim: int, splits: int, min_pages: int = MIN_SPLIT_PAGES,
              warps: Optional[int] = None):
    """The gfx950 merge for (head_dim, splits), built once; `warps` defaults
    to merge_warps(splits)."""
    warps = merge_warps(splits) if warps is None else warps
    key = ("merge", head_dim, splits, min_pages, warps)
    k = _KERNELS.get(key)
    if k is None:
        tag = "" if min_pages == MIN_SPLIT_PAGES else f"_minpages{min_pages}"
        tag += "" if warps == merge_warps(splits) else f"_w{warps}"
        k = _KERNELS[key] = merge_kernel(f"paged_merge_d{head_dim}_s{splits}{tag}_cdna",
                                         head_dim=head_dim, splits=splits, min_pages=min_pages,
                                         warps=warps)
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


__all__ = ["decode_kernel", "kernel_for", "PAGE", "Q_TILE", "HEAD_DIMS", "ABLATIONS",
           "target_workgroups", "merge_warps", "MAX_SPLITS"]
