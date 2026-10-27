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
#: The floor where the merge reads every split (merge_reads_all, up to 64
#: splits): with no rounds to skip, a short sequence is better cut finer
#: -- one request of 1k tokens on 4-8 kv heads 1.07-1.12x four pages' floor.
MIN_SPLIT_PAGES_ALL = 2
#: A split at least this long walks two pages a step in a two-page build
#: (decode_kernel(ppstep=2)); a shorter one a page a step, which was up to 2%
#: faster on four-page splits and 10-20% on two-page ones.
PP2_MIN_PAGES = 8
#: Sequences of at most this many pages take MIN_SPLIT_PAGES_ALL whatever the
#: plan (floor_pages).
SHORT_PAGES = 128
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


def pages_per_step(pairs: int) -> int:
    """Pages a decode workgroup takes a step (decode_kernel's `ppstep`) with
    `pairs` (request, kv head) pairs: two where the plan puts about one
    workgroup on a CU (under 64 pairs, and 256-511), where a lone wave's per-page
    work is the walk's time -- 1.02-1.06x from 16k tokens on, within 2% of
    one at 1k; one where it puts two (512), where two pages a step were up
    to 3% slower; and one from 512 pairs, two workgroups a CU again with no
    split, where two were no faster and 1% slower at 1k tokens."""
    return 2 if pairs < 64 or 256 <= pairs < 512 else 1


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
                  min_pages: Optional[int] = None, ppstep: int = 1,
                  fence_qk: bool = True, short_pages: int = SHORT_PAGES):
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

    `ppstep` pages a step (1 or 2): Sᵀ is that many pages tall and a step's
    softmax, waits and barriers are shared by its pages (pages_per_step).
    `fence_qk=False` drops the scheduling fence after two pages' QK, for an
    A/B only: without it this toolchain's build computes wrong scores.

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
    T_O = hk.RT(fp32, D, Q_TILE, "col")        # Oᵀ, [d, q]
    if ppstep not in (1, 2):
        raise ValueError(f"ppstep={ppstep}: 1 or 2 pages a step")
    if min_pages is None:
        min_pages = min_pages_for(n_split)
    ST = hk.ST(bf16, PAGE, D)

    @hk.func
    def walk(q, k_cache, v_cache, block_table, seq_lens, finish):
        g = hk.group(1)
        Ks = hk.lds.alloc(ST, count=stages * ppstep, name="Ks", shape=K_SHAPE)
        Vs = hk.lds.alloc(ST, count=stages * ppstep, name="Vs", shape=V_SHAPE)
        kv_head = hk.arch.block_idx.y
        req = hk.arch.block_idx.z
        # Workgroup w runs on XCD w % 8, and with 8 or more splits the split
        # index alone picked it: a short sequence's first splits, the ones
        # with pages, all landed on the first XCDs. Turned by the pair, they
        # spread.
        split = (hk.arch.block_idx.x + kv_head + req) % n_split if n_split > 1 else hk.arch.block_idx.x

        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
        n_pages = hk.cdiv(seq_len, PAGE)
        per = hk.max(hk.cdiv(n_pages, n_split), floor_pages(n_pages, min_pages, short_pages))
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

        def run(PP):
            """The split's walk at PP pages a step, and its finish."""
            T_S = hk.RT(fp32, PP * PAGE, Q_TILE, "col")     # Sᵀ, [kv, q]
            T_P = hk.RT(bf16, PP * PAGE, Q_TILE, "col", frag=(32, 16, 4))   # Pᵀ, the operand
            T_VEC = hk.row_vec(T_S)                    # one value per query

            def copy(t, slot, pages):
                # Step t: its pages t * PP + p into the slot's buffers.
                for p in range(PP):
                    rows = rows_of(t * PP + p)
                    g.load(Ks[slot * PP + p], k_cache[hk.elem_coord(pages[p], kv_head, 0, 0)],
                           wait=False, rows=rows, nontemporal=True)
                    g.load(Vs[slot * PP + p], v_cache[hk.elem_coord(pages[p], kv_head, 0, 0)],
                           wait=False, rows=rows, nontemporal=True)

            def pages_of(t):
                return [page_of(t * PP + p) for p in range(PP)]

            qr = q[hk.elem_coord(req, kv_head, 0, 0)].load(T_Q, bounded=True)
            krs = [hk.zeros(T_K) for _ in range(PP)]
            vrs = [hk.zeros(T_V) for _ in range(PP)]
            S = hk.zeros(T_S)
            P = hk.zeros(T_P)
            O = hk.zeros(T_O)
            m = hk.neg_infty(T_VEC)
            m_old = hk.neg_infty(T_VEC)
            ell = hk.zeros(T_VEC)

            def math_page(slot, t):
                for p in range(PP):
                    Ks[slot * PP + p].load(T_K, wait=False, into=krs[p])
                    Vs[slot * PP + p].load(T_V, wait=False, into=vrs[p])
                hk.lds.wait_for(*krs)
                if "math" in ablate:
                    hk.lds.wait_for(*vrs)
                    return
                S.zero_()
                if PP == 1:
                    S.__iadd__(krs[0] @ qr.T)
                else:
                    for p in range(PP):
                        Sp = S.subtile(PAGE, p)
                        Sp += krs[p] @ qr.T
                    if fence_qk:
                        # Nothing crosses the end of QK. Left free, LLVM
                        # threads the softmax's reads of the AGPR scores
                        # through the two pages' MFMA chains and some scores
                        # come out wrong (2-4e-2 off on a split of 128
                        # tokens, a different answer from one build to the
                        # next); no wait or barrier anywhere else mends it.
                        hk.arch.sched_barrier()
                S.__imul__(scale_l2e)
                for p in range(PP):
                    rows = rows_of(t * PP + p)
                    if rows < PAGE:
                        if PP == 1:
                            S.lower_fill_(rows, NEG_INF)
                        else:
                            S.subtile(PAGE, p).lower_fill_(rows, NEG_INF)
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
                hk.lds.wait_for(*vrs)
                if PP == 1:
                    O.__iadd__(vrs[0].T @ P)
                else:
                    for p in range(PP):
                        O.__iadd__(vrs[p].T @ P.subtile(PAGE, p))

            # The page index a copy needs is read a page ahead of it, so the
            # copy does not wait out a load of the table.
            n_steps = hk.cdiv(n, PP)
            for j in range(stages - 1):
                copy(j, j, pages_of(j))
            # The next step's page indices, one loop-carried scalar a page.
            nxt0 = page_of((stages - 1) * PP)
            nxt1 = page_of((stages - 1) * PP + 1) if PP == 2 else nxt0
            for it in range(hk.cdiv(n_steps, stages)):
                for j in range(stages):
                    t = it * stages + j
                    # Step t-1's buffers: their reads retired in the last
                    # math, and a barrier after them before the copy
                    # overwrites them.
                    hk.arch.barrier()
                    pages = [nxt0, nxt1][:PP]
                    nxt0 = page_of((t + stages) * PP)
                    if PP == 2:
                        nxt1 = page_of((t + stages) * PP + 1)
                    copy(t + stages - 1, (j + stages - 1) % stages, pages)
                    g.wait_for(*[Ks[j * PP + p] for p in range(PP)],
                               *[Vs[j * PP + p] for p in range(PP)])
                    hk.arch.barrier()
                    math_page(j, t)
            # The ring's last copies are of pages past the split: nothing,
            # but in flight into LDS.
            hk.arch.vm_wait(0)
            if mode == "fused" and not merge_reads_all(n_split):
                # Past a round a merge warp (merge_reads_all) the merge reads
                # the rounds of MERGE_CHUNK splits that hold the first
                # `active` (one round for a padded request); a split past
                # them has nothing to write.
                active = hk.cdiv(n_pages, per)
                read = hk.max(hk.cdiv(active, MERGE_CHUNK), 1) * MERGE_CHUNK
                if split < read:
                    finish(O, m, ell, T_VEC, split, kv_head, req)
            else:
                finish(O, m, ell, T_VEC, split, kv_head, req)

        def walk_split():
            if ppstep == 1:
                run(1)
            else:
                # Two pages a step where the split is long enough to share a
                # step's softmax and waits; a short split (a short sequence
                # cut many ways) walks a page a step.
                if hk.uniform(n >= PP2_MIN_PAGES):
                    run(2)
                else:
                    run(1)

        if merge_reads_all(n_split):
            walk_split()
        else:
            # A plan of more than a round a merge warp is for long
            # sequences; on a short one most of its splits are empty, and
            # those skip the query load and the copies.
            if hk.uniform(n > 0):
                walk_split()
            else:
                T_VEC = hk.row_vec(hk.RT(fp32, PAGE, Q_TILE, "col"))
                O = hk.zeros(T_O)
                m = hk.neg_infty(T_VEC)
                ell = hk.zeros(T_VEC)
                active = hk.cdiv(n_pages, per)
                read = hk.max(hk.cdiv(active, MERGE_CHUNK), 1) * MERGE_CHUNK
                if mode != "fused" or split < read:
                    finish(O, m, ell, T_VEC, split, kv_head, req)

    def normalise(O, ell, T_VEC):
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
            def finish(O, m, ell, T_VEC, split, kv_head, req):
                normalise(O, ell, T_VEC)
                o[hk.elem_coord(req, kv_head, 0, 0)].store(hk.transpose(O), bounded=True)
            walk(q, k_cache, v_cache, block_table, seq_lens, finish)
    elif mode == "split":
        def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16],
                 v_cache: hk.Tensor[bf16], block_table: hk.Tensor[i32],
                 seq_lens: hk.Tensor[i32], o_part: hk.Tensor[fp32],
                 ml_part: hk.Tensor[fp32]):
            def finish(O, m, ell, T_VEC, split, kv_head, req):
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
            def finish(O, m, ell, T_VEC, split, kv_head, req):
                if n_split == 1:
                    normalise(O, ell, T_VEC)
                    out[hk.elem_coord(req, kv_head, 0, 0)].store(
                        hk.transpose(O), bounded=True)
                    return
                # Normalised, its values are a convex mix of V's, so bf16
                # costs the partial what bf16 costs the output; an empty
                # split has l = 0 and O = 0, and the merge weighs it zero.
                row = req * n_split + split
                ml_part[hk.elem_coord(row, kv_head, 0, 0)] = m
                ml_part[hk.elem_coord(row, kv_head, 1, 0)] = ell
                normalise(O, ell, T_VEC)
                o_part[hk.elem_coord(row, kv_head, 0, 0)].store(
                    hk.transpose(O), bounded=True)
            walk(q, k_cache, v_cache, block_table, seq_lens, finish)

    body.__name__ = name
    return hk.kernel(body, arch=arch, warps=1, name=name,
                     grid=lambda p: (n_split, p.q.depth, p.q.batch))


#: Splits the gfx950 merge reads a round.
MERGE_CHUNK = 8
#: The most warps a merge workgroup folds with: 8 was 3-10% faster than 16
#: from 4k tokens on at 256 splits (combining fewer folds), and no slower
#: below.
MERGE_WARPS = 8


def merge_reads_all(splits: int) -> bool:
    """Whether the merge (with merge_warps(splits) warps) reads every split:
    up to a round a warp it does, without waiting on the sequence's length
    first -- 0.6-1 us of a one-request decode -- so the walk writes even an
    empty split's partial (zero, weightless); past that it reads only the
    rounds that hold pages, and only those are written."""
    return splits <= MERGE_CHUNK * merge_warps(splits)


def floor_pages(n_pages, min_pages: int, short_pages: int):
    """The page floor a split of a sequence of `n_pages` takes (walk and
    merge alike): `min_pages`, but MIN_SPLIT_PAGES_ALL for a sequence of at
    most `short_pages` -- a plan of many splits for long sequences walked a
    short one 4 pages a split, 1.2 us longer than 2 at 1-2k tokens; one
    request on one kv head at 2-4k tokens is 1.15-1.4x main before the
    plan went to 256 splits, where it was 0.93-1.27x."""
    if not short_pages or min_pages <= MIN_SPLIT_PAGES_ALL:
        return min_pages
    short = hk.min(hk.max(short_pages + 1 - n_pages, 0), 1)
    return min_pages - (min_pages - MIN_SPLIT_PAGES_ALL) * short


def min_pages_for(splits: int) -> int:
    """The fewest pages a split of `splits` takes (walk and merge agree):
    MIN_SPLIT_PAGES_ALL where the merge reads every split, MIN_SPLIT_PAGES
    past that, where fewer splits that saw pages are fewer rounds read."""
    return MIN_SPLIT_PAGES_ALL if merge_reads_all(splits) else MIN_SPLIT_PAGES


def merge_warps(splits: int) -> int:
    """Warps of the merge for `splits` splits: about a round of MERGE_CHUNK
    each, so the fold is a round or two long whatever the split count."""
    return min(MERGE_WARPS, max(1, splits // MERGE_CHUNK))


def merge_kernel(name: str, *, head_dim: int = 128, splits: int = 8,
                 arch: Optional[str] = None, min_pages: Optional[int] = None,
                 warps: int = 1, short_pages: int = SHORT_PAGES):
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
    if min_pages is None:
        min_pages = min_pages_for(splits)
    run = D // 64
    if D % 64 or run not in (1, 2):
        raise ValueError(f"head_dim={D}: a strip of 64 or 128")
    T = hk.RS(fp32, 1, D, run=run)
    CV = hk.col_vec(T)

    chunk = min(MERGE_CHUNK, splits)
    all_rounds = splits <= chunk * warps
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
        # past them would be all empty splits, weightless, and is not run --
        # but up to a round a warp each warp reads its round without waiting
        # for the length (an empty round is the walk's zeros), and the length
        # only decides whether the warps' folds need combining.
        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
        n_pages = hk.cdiv(seq_len, PAGE)
        active = hk.cdiv(n_pages, hk.max(hk.cdiv(n_pages, splits),
                                         floor_pages(n_pages, min_pages, short_pages)))
        seen = hk.max(hk.cdiv(active, chunk), 1)
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
        rounds = splits // chunk if all_rounds else seen
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
            if hk.uniform(seen > 1):
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
                    # Only warps that folded a round hold anything: a short
                    # sequence on many warps sums one or two strips, not
                    # fifteen element-by-element reads.
                    if hk.uniform(seen > w):
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


def merge_for(head_dim: int, splits: int, min_pages: Optional[int] = None,
              warps: Optional[int] = None, short_pages: int = SHORT_PAGES):
    """The gfx950 merge for (head_dim, splits), built once; `warps` defaults
    to merge_warps(splits), `min_pages` to min_pages_for(splits)."""
    warps = merge_warps(splits) if warps is None else warps
    min_pages = min_pages_for(splits) if min_pages is None else min_pages
    key = ("merge", head_dim, splits, min_pages, warps, short_pages)
    k = _KERNELS.get(key)
    if k is None:
        tag = "" if min_pages == min_pages_for(splits) else f"_minpages{min_pages}"
        tag += "" if warps == merge_warps(splits) else f"_w{warps}"
        tag += "" if short_pages == SHORT_PAGES else f"_short{short_pages}"
        k = _KERNELS[key] = merge_kernel(f"paged_merge_d{head_dim}_s{splits}{tag}_cdna",
                                         head_dim=head_dim, splits=splits, min_pages=min_pages,
                                         warps=warps, short_pages=short_pages)
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
           "target_workgroups", "pages_per_step", "merge_warps", "merge_reads_all",
           "MAX_SPLITS"]
