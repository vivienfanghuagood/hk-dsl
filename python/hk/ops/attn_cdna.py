"""Flash attention forward on gfx950 (CDNA4, MI355X), written in the DSL.

The schedule is HipKittens' CDNA4 GQA forward (kernels/cdna4/attn/gqa and
gqa_causal) written in hk, on hk's own operand layout, (B, H, N, D) -- the
one torch's scaled_dot_product_attention takes.

Eight warps, 32 query rows each, so a workgroup owns 256 rows; the KV axis is
walked in 64-row blocks, K and V double-buffered in LDS by gfx950's DMA
(`g.load(wait=False)`, the vmcnts derived by hk-lds-pipeline). Each step of
the walk is four phases a barrier apart, two of math and two of memory:

    QK(t+1) and the end of softmax(t)  |  copy K(t+3), read V(t), mask S(t+1)
    PV(t) and the start of softmax(t+1) |  copy V(t+2), read K(t+2)

and the two halves of the workgroup run a barrier apart (`hk.arch.pingpong`),
so on every SIMD one wave is at the matrix core while the other is at
memory.

What made it expressible on gfx950, where the softmax's P comes out of the
MFMA as an accumulator and P·V wants it as an operand:

  * **The scores are kept transposed.** Sᵀ = K·Qᵀ is a 64x32 accumulator in
    32x32 fragments (the 32x32x16 MFMA), so the softmax's statistics are one
    value per query -- a row vector of Sᵀ -- and Oᵀ (D x 32) takes them with
    `mul_col` (the final 1/l too: one division a query, not one an
    element).
  * **Pᵀ is read in place as the operand** (`Tile.as_operand`). A 32x32
    accumulator's lane holds four rows at a time of its column, which is a
    16x32 operand fragment with a lane run of 4 instead of 8: the reduction
    index (the KV position) runs down the rows in another order. Oᵀ += Vᵀ·Pᵀ
    then reads V out of LDS with the same permutation (hk-fragments gives
    the MMA's other operand the same stride), and the product is unchanged.
    No register moves between the softmax and the second MMA.

Where this differs from HipKittens' kernel:

  * **The rescale.** HipKittens rescales O by exp2(m_old - m_new) after the
    first quarter of P(t)·V and before the other three, so when the max does
    jump, three quarters of block t are added at the wrong scale. Here the
    decision (the max grew by more than 8 in log2 units, one vote across the
    wave) and the new max are taken without a branch in the PV phase, and O
    is rescaled at the top of the next math phase, after all of P(t)·V and
    before P(t+1)·V. l takes the factor at once; it is one value a lane.
  * **Causal.** A warp masks a block only when it reaches past one of its
    queries, with one element-wise triu over the whole block; the last two
    steps, wholly past the first half's queries, are memory only for those
    warps (`skip`). And a workgroup walks two query blocks, the longest and
    the shortest left -- (nqb - 1 - y, y) -- so every workgroup does the same
    work (`pairs`): with longest-first order alone, a grid of few workgroups a
    batch ended on a round of long tiles with most of the machine idle
    (Llama-3-70B at TP8, 8 query heads, 8 x 2048: 1.30x).
  * **Any N.** Q rows past N read zeros and are not stored; a KV block that
    runs past N_kv is masked with -inf (the DMA reads zeros there); an odd
    block count gets one fully masked block, so the walk is whole steps.
  * **(B, H, N, D)**, contiguous rows of one head, where HipKittens takes
    (B, N, H, D).
  * **`mapping="spatial"`** (arXiv 2511.02132; AITER's remap_workgroup_spatial).
    The default grid deals a query block of every head before the next block
    (HipKittens' numbering), so each XCD holds H_KV / 8 KV heads' streams at
    once; with many KV heads and long sequences they drift apart and the
    XCD's 4 MB L2 stops serving one workgroup's K/V to the next. "spatial"
    gives each XCD a contiguous run of (batch, KV head) clusters, one cluster
    at a time, so the workgroups resident on an XCD read one K/V stream.
    The kernel is the same: hk.dsl.placement finds the cluster key, (batch,
    KV head), in the traced loads and installs the numbering at entry; each
    workgroup still computes the same tile the same way, so the output is
    bitwise the default's.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .. import dsl as hk
from ..dsl import placement as _pl
from ..types import bf16, fp32, i32

NEG_INF = float("-inf")

#: Query rows a warp owns, warps a workgroup, query rows a workgroup.
Q_ROWS, WARPS = 32, 8
Q_TILE = Q_ROWS * WARPS
#: The KV block.
KV_BLOCK = 64
#: The paged cache's page (hk.ops.paged.KV_BLOCK): two to a KV block.
PAGE = 32
#: Head dims with a schedule.
HEAD_DIMS = (64, 128)
#: Workgroup numberings: HipKittens' (a query block of every head, then the
#: next block), and one (batch, KV head) cluster an XCD at a time.
MAPPINGS = ("block_first", "spatial")


#: Workgroups an XCD of an MI355X holds at once: one a CU (8 warps, 232
#: VGPRs), 32 CUs an XCD.
RESIDENT_PER_XCD = 32


def mapping_for(batch: int, h: int, h_kv: int, n_q: int, causal: bool,
                resident_per_xcd: int = RESIDENT_PER_XCD) -> str:
    """The numbering hk.ops.attn routes a call to: hk.dsl.placement.choose on
    the dense kernel's own analysis (its cluster key, its non-uniform walk under
    causal) at this grid -- "spatial" where the default would hold five or more
    KV streams on an XCD at once, except a causal grid with fewer than twelve
    rounds of workgroups an XCD. The thresholds are the ones an MI355X sweep
    measured on this kernel (tools/hk-bench/attn_mapping.py, 2026-10-26): +2-5% at
    40 KV heads, +5-8% at 64, +10-18% at 128 (N 8K-128K), the gain mostly clock;
    -10% for a short causal grid, whose last cluster ends on its longest tile."""
    xcds = _pl.xcd_count("gfx950")
    an = _analysis(causal)
    plan = _pl.make_plan(_pl.Placement.cluster(), an, xcds) if an.kind == "cluster" else None
    grid = (h, -(-n_q // Q_TILE), batch)
    c = _pl.choose(an, plan, grid, {"g.k.depth()": h_kv}, xcds, resident_per_xcd)
    return "spatial" if c.placed else "block_first"


_ANALYSES: Dict[bool, Any] = {}


def _analysis(causal: bool):
    """hk.dsl.placement.analyze of the dense kernel (head_dim 128; the key and
    the walk do not depend on it), once a process."""
    if causal not in _ANALYSES:
        from ..compiler import mlir as M  # noqa: PLC0415

        tr = attn_kernel(f"attn_fwd_analysis_{int(causal)}", causal=causal,
                         arch="gfx950").trace()
        with M.context():
            _ANALYSES[causal] = _pl.analyze(tr.module, tr.name)
    return _ANALYSES[causal]


def _pairs(n_mma: int, groups, sync_id: int) -> None:
    """Interleave `n_mma` MFMAs with other instructions: `groups` is a list
    of (count, class, size) -- `count` MFMAs, each followed by `size`
    instructions of `class`. HipKittens' sched_barrier_pairs."""
    used = 0
    for count, cls, size in groups:
        for _ in range(count):
            hk.arch.sched_group_barrier("mma", 1, sync_id)
            if size:
                hk.arch.sched_group_barrier(cls, size, sync_id)
        used += count
    if used != n_mma:
        raise ValueError(f"the interleave places {used} MFMAs, the phase has {n_mma}")


def attn_kernel(name: str, *, head_dim: int = 128, causal: bool = False,
                scale: Optional[float] = None, setprio: bool = True,
                lazy: float = 8.0, sched: bool = True, lpt: bool = True,
                skip: bool = True, _nomask: bool = False, paged: bool = False,
                mapping: str = "block_first", arch: Optional[str] = None,
                pairs: bool = False, kv_split: bool = False, acc: str = "vgpr",
                mma_fence: bool = True):
    """`acc="agpr"` holds O in AGPRs (hk.arch.agpr): bitwise the default's,
    measured slower (an asm MFMA LLVM cannot schedule), so off by default.
    `mma_fence=False` turns hk-mma-fence off for an A/B (hk.arch.mma_fence).

    `lazy`: the running max is taken as stale until a block's max exceeds
    it by more than this (log2 units), so P stays below 2^lazy and O is
    rescaled only when the max really moves (FlashAttention 4's threshold,
    HipKittens' 8). 0 rescales whenever it grows at all.

    `sched` keeps HipKittens' interleave of MFMAs with VALU and exp2 in the
    math phases; `setprio` its s_setprio around the PV phase; `lpt` reverses
    the query blocks under `causal`, so the longest workgroups start first;
    `skip` drops, under `causal`, the last two steps' math for the warps
    whose queries they are wholly past. `_nomask` (wrong answers) is for
    timing what the masks cost.

    `paged` is the same schedule over a paged KV cache, for chunked prefill
    and prefix caching (hk.ops.paged_prefill's contract): q and o are the
    engine's (tokens, heads * D) rows, K and V (blocks, kv heads, PAGE, D)
    pages through `block_table`, and a workgroup is one entry of the
    (request, query tile) work list `tiles`; a request's new tokens are the
    last q_len of its sequence, causal from the cached prefix on. A 64-row
    KV block is two pages, each its own copy and LDS buffer, read into the
    two 32-row halves of the block's registers.

    `pairs` (dense causal, the default numbering; kernel_for turns it on): a
    workgroup walks query blocks nqb - 1 - y and then y, one after the other,
    so every workgroup walks nqb + 1 blocks' worth (the middle block of an odd
    count alone); the grid's y is cdiv(nqb, 2). Each block is computed as
    without it, so the output is bitwise the same.

    `mapping`: the workgroup numbering (MAPPINGS; the module's docstring),
    installed by hk.dsl.placement: "spatial" is hk.Placement.cluster() -- the
    analysis's key (batch, KV head) -- and, paged, the key KV head given
    explicitly (a page's address goes through the block table, which the
    analysis cannot see through), so a cluster is (KV head, the work list in
    order), whose requests are contiguous runs of it.
    """
    D = head_dim
    if acc not in ("vgpr", "agpr"):
        raise ValueError(f"acc={acc!r}: 'vgpr' or 'agpr'")
    if mapping not in MAPPINGS:
        raise ValueError(f"mapping={mapping!r}: one of {MAPPINGS}")
    placement = None
    if mapping == "spatial":
        placement = _pl.Placement.cluster(
            key=(_pl.Digit.of("x", "mod", "k_cache.depth"),) if paged else None)
    if D not in HEAD_DIMS:
        raise ValueError(f"head_dim={D}: gfx950's attention and paged prefill have "
                         f"a schedule for {HEAD_DIMS} only")
    scale_l2e = (scale if scale is not None else D ** -0.5) * math.log2(math.e)
    half = KV_BLOCK // 2
    # MFMAs a phase issues: QK is (64 x D) . (D x 32) in 32x32x16 steps, PV
    # (D x 64) . (64 x 32); both are 2 * D / 16.
    n_mma = 2 * D // 16

    if paged and not causal:
        raise ValueError("the paged prefill is causal")
    if kv_split and (not paged or mapping != "block_first"):
        raise ValueError("kv_split: the paged prefill on the default numbering")
    # Pairs are the dense causal kernel's, on the default numbering (the
    # spatial one is routed on measurements of the single-block walk).
    pairs = bool(pairs) and causal and not paged and mapping == "block_first"
    T_Q32 = hk.RT(fp32, Q_ROWS, D, "row")      # this warp's Q, scaled in fp32
    # A K or V block in registers, [kv, d]; paged, two pages of it.
    kv_rows = PAGE if paged else KV_BLOCK
    T_K = hk.RT(bf16, kv_rows, D, "row")
    T_V = hk.RT(bf16, kv_rows, D, "col")
    T_S = hk.RT(fp32, KV_BLOCK, Q_ROWS, "col", frag=(32, 32))     # Sᵀ, [kv, q]
    T_P = hk.RT(bf16, KV_BLOCK, Q_ROWS, "col", frag=(16, 32, 4))  # Pᵀ as an operand
    T_O = hk.RT(fp32, D, Q_ROWS, "col", frag=(32, 32))            # Oᵀ, [d, q]
    T_VEC = hk.row_vec(T_S)                    # one value per query
    ST = hk.ST(bf16, kv_rows, D)
    # Buffers a block takes in LDS: one, or its pages.
    nbuf = KV_BLOCK // kv_rows

    def alloc():
        return (hk.lds.alloc(ST, count=2 * nbuf, name="Ks"),
                hk.lds.alloc(ST, count=2 * nbuf, name="Vs"))

    def walk(q, k, v, o, block_table, seq_lens, q_start, tiles, second=0, lds=None,
             o_part=None, ml_part=None):
        """`second`: under `pairs`, 0 for the pair's long block, 1 for its
        short one; `lds`: K's and V's buffers, allocated once for both."""
        g = hk.group(WARPS)
        Ks, Vs = lds if lds is not None else alloc()

        if not paged:
            n_q, n_kv = q.rows, k.rows
            b = hk.arch.block_idx.z
        x = hk.arch.block_idx.x
        # GQA: workgroup x of a query block takes KV head x % H_KV, so
        # consecutive workgroups -- dealt round robin to the XCDs -- read
        # different KV heads, and the query heads that share one KV head land
        # on one XCD and share its L2 (HipKittens' numbering).
        h_kv = k.depth
        group = (q.cols // D if paged else q.depth) // h_kv
        head_kv = x % h_kv
        head = head_kv * group + x // h_kv
        if paged:
            # A work list entry: the request and its query tile. Query i of
            # the request's new tokens sits at position ctx + i.
            t_ = hk.arch.block_idx.y
            req = tiles[hk.elem_coord(0, 0, t_, 0)].load_scalar(invariant=True)
            qb = tiles[hk.elem_coord(0, 0, t_, 1)].load_scalar(invariant=True)
            tok0 = q_start[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
            n_q = q_start[hk.elem_coord(0, 0, 0, req + 1)].load_scalar(invariant=True) - tok0
            seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar(invariant=True)
            ctx = seq_len - n_q
            last_col = block_table.cols - 1
            if kv_split:
                # The entry's KV blocks [blo, bhi) (bhi -1: to the tile's
                # last query, the diagonal included) and its partial's slot
                # (-1: the answer, straight to o).
                blo = tiles[hk.elem_coord(0, 0, t_, 2)].load_scalar(invariant=True)
                bhi = tiles[hk.elem_coord(0, 0, t_, 3)].load_scalar(invariant=True)
                slot = tiles[hk.elem_coord(0, 0, t_, 4)].load_scalar(invariant=True)
        else:
            qb = hk.arch.block_idx.y
            if pairs:
                nqb = hk.cdiv(n_q, Q_TILE)
                qb = nqb - 1 - qb - second * (nqb - 1 - 2 * qb)
            elif causal and lpt:
                qb = hk.cdiv(n_q, Q_TILE) - 1 - qb
            ctx = 0
        q0 = qb * Q_TILE
        wid = hk.arch.warp_id()
        q_row = q0 + wid * Q_ROWS

        # KV blocks this workgroup walks: all of them, or under causal up to
        # its last query; then a whole number of steps of two, and at least
        # four (the blocks past N_kv are masked), so the loop below runs at
        # least once and the compiler can see it: with a path around the
        # loop, values for the epilogue were held across it, and spilled.
        blo_ = 0
        if paged:
            # Blocks past the sequence copy nothing and are past every query.
            n_blk = hk.cdiv(ctx + hk.min(q0 + Q_TILE, n_q), KV_BLOCK)
            seq_end = seq_len
            if kv_split:
                # A split ends at bhi: its copies stop there (seq_end) and the
                # blocks past it are masked (kv_end); the last split is the
                # unsplit walk from blo.
                last = hk.min(hk.max(-bhi, 0), 1)
                hi = last * n_blk + (1 - last) * bhi
                seq_end = last * seq_len + (1 - last) * hi * KV_BLOCK
                kv_end = last * (1 << 30) + (1 - last) * (hi * KV_BLOCK - ctx)
                n_blk = hi - blo
                blo_ = blo
        else:
            n_blk = hk.cdiv(n_kv, KV_BLOCK)
            if causal:
                n_blk = hk.min(n_blk, hk.cdiv(hk.min(q0 + Q_TILE, n_q), KV_BLOCK))
        n_iter = hk.max(hk.cdiv(n_blk, 2) - 1, 1)
        n_blk = 2 * n_iter + 2

        # A page is 64 D bytes, a warp's copy 1 KB: at D = 64 half the
        # warps cover it, and the other half copies the same bytes again
        # (identical LDS writes; the second read is the L2's).
        gp = hk.group(min(WARPS, PAGE * D * 2 // 1024)) if paged else g

        def copy_pages(dst, src, t, par):
            # A page past the sequence is copied with no rows -- zeros, and
            # no address followed: the table's unused columns hold anything.
            for p in range(nbuf):
                pos = (blo_ + t) * nbuf + p
                page = block_table[hk.elem_coord(
                    0, 0, req, hk.min(pos, last_col))].load_scalar(invariant=True)
                rows = hk.min(hk.max(seq_end - pos * PAGE, 0), PAGE)
                gp.load(dst[par * nbuf + p], src[hk.elem_coord(page, head_kv, 0, 0)],
                        wait=False, rows=rows)

        def copy_k(t, par):
            if paged:
                copy_pages(Ks, k, t, par)
                return
            g.load(Ks[par], k[hk.elem_coord(b, head_kv, t * KV_BLOCK, 0)], wait=False)

        def copy_v(t, par):
            if paged:
                copy_pages(Vs, v, t, par)
                return
            g.load(Vs[par], v[hk.elem_coord(b, head_kv, t * KV_BLOCK, 0)], wait=False)

        def bufs(st, par):
            return [st[par * nbuf + p] for p in range(nbuf)]

        krs = [hk.zeros(T_K) for _ in range(nbuf)]
        vrs = [hk.zeros(T_V) for _ in range(nbuf)]
        # Two blocks of scores in flight, each one tile -- one reduction and
        # one conversion over 64 rows -- whose 32-row halves (subtiles, the
        # same registers) take their exp2 in different phases.
        S = [hk.zeros(T_S) for _ in range(2)]
        Sh = [[S[bb].subtile(half, h) for h in range(2)] for bb in range(2)]
        P_op = hk.zeros(T_P)
        O = hk.zeros(T_O)
        if not mma_fence:
            hk.arch.mma_fence(False)
        if acc == "agpr":
            hk.arch.agpr(O)
        m = hk.zeros(T_VEC)          # the running max S is taken against
        # The new running max, and once it is taken, exp2(m_old - m): the
        # rescale O takes at the top of the next math phase.
        m_new = hk.full(T_VEC, 1.0)
        sc = m_new
        ell = hk.zeros(T_VEC)

        def read_k(par, wait=True):
            for p in range(nbuf):
                Ks[par * nbuf + p].load(T_K, wait=False, into=krs[p])
            if wait:
                hk.lds.wait_for(*krs)

        def read_v(par, wait=True):
            for p in range(nbuf):
                Vs[par * nbuf + p].load(T_V, wait=False, into=vrs[p])
            if wait:
                hk.lds.wait_for(*vrs)

        def qk(bb):
            S[bb].zero_()
            if nbuf == 1:
                S[bb] += krs[0] @ qb_.T
            else:
                for p in range(nbuf):
                    Sp = S[bb].subtile(kv_rows, p)
                    Sp += krs[p] @ qb_.T

        def mask(t, bb):
            # Sᵀ rows are KV positions and columns queries, so the causal
            # mask (kv > q) is triu and the KV tail rows a lower fill. Each
            # behind a wave-uniform branch: most blocks need neither.
            kv0 = (blo_ + t) * KV_BLOCK
            if paged:
                kv0 = kv0 - ctx
            if causal and not _nomask:
                # One branch, one path: a block that reaches past any of
                # this warp's queries takes the element-wise triu whole,
                # which is -inf throughout a block wholly past them. With a
                # path for each case (all -inf, or triu on one 32-row half)
                # the tile came out of the branch in registers of its own,
                # copied into on every step that masks nothing.
                if q_row < kv0 + KV_BLOCK:
                    S[bb].triu_(q_row - kv0, NEG_INF)
            # The KV tail. Under causal (N_kv == N_q) a row past N_kv is past
            # every query too, so the mask above has already taken it.
            if not causal and not _nomask:
                if kv0 + KV_BLOCK > n_kv:
                    S[bb].lower_fill_(n_kv - kv0, NEG_INF)
            # A split's padding past bhi: real keys, another split's.
            if kv_split and not _nomask:
                if kv0 + KV_BLOCK > kv_end:
                    S[bb].lower_fill_(kv_end - kv0, NEG_INF)

        def softmax_first(bb):
            m.max_(S[bb].max(axis=0))        # m was -inf
            S[bb].__isub__(m)
            Sh[bb][0].exp2_()

        # The pins below hold a phase's results to its phase: their next use
        # is a phase away, and LLVM sinks a value to its use, past the
        # barrier, where it lands in the other phase's math (sched_barrier
        # fences the scheduler, not the other passes).

        def softmax_start(bb):
            """The new running max and, with no branch, whether to move to
            it; S - m, and exp2 of its first half."""
            m.copy(into=m_new)
            m_new.max_(S[bb].max(axis=0))
            grow = m_new.any_above(m, lazy)
            m_new.__isub__(m)               # the step up
            m_new.__imul__(grow)            # 0 unless the max moves
            m.__iadd__(m_new)
            hk.math.mul(m_new, -1.0, into=sc)
            sc.exp2_()                      # 1 unless the max moves
            hk.arch.pin(m, sc, ell)
            ell.__imul__(sc)
            S[bb].__isub__(m)
            Sh[bb][0].exp2_()
            hk.arch.pin(S[bb])
            return grow

        def softmax_end(bb):
            """exp2 of the block's second half, its sum into l, and its bf16
            copy read as the PV operand (HipKittens' split: the PV phase,
            which also takes the max, carries the less VALU)."""
            Sh[bb][1].exp2_()
            ell.__iadd__(S[bb].sum(axis=0))
            S[bb].as_operand(bf16, into=P_op)
            hk.arch.pin(P_op)

        def pv():
            if nbuf == 1:
                O.__iadd__(vrs[0].T @ P_op)
            else:
                for p in range(nbuf):
                    O.__iadd__(vrs[p].T @ P_op.subtile(kv_rows, p))

        def rescale(grow):
            if grow > 0:
                O.__imul__(hk.as_vec(sc, T_O))

        def end_phase():
            hk.arch.sched_barrier()
            hk.arch.barrier()
            hk.arch.sched_barrier()

        # Every math phase's interleave gets its own sync id, as HipKittens'
        # does: groups that share one are chained in program order across
        # phases, and reusing two ids for every phase of the loop body gave
        # a wrong answer on the device.
        sync = iter(range(1, 1 << 30))

        def math_qk(t1, b1, b0, grow):
            """QK(t+1) into S[b1] and the end of softmax(t) on S[b0]."""
            rescale(grow)
            qk(b1)
            softmax_end(b0)
            if sched in (True, "qk"):
                _pairs(n_mma, [(6, "trans", 3), (n_mma - 6, "valu", 5)], next(sync))
            end_phase()

        def math_pv(b1):
            """PV(t), and the start of softmax(t+1) on S[b1]."""
            if setprio:
                hk.arch.setprio(1)
            pv()
            grow = softmax_start(b1)
            if sched in (True, "pv"):
                _pairs(n_mma, [(n_mma - 6, "valu", 5), (6, "trans", 3)], next(sync))
            if setprio:
                hk.arch.setprio(0)
            end_phase()
            return grow

        # `skip`: under causal, the blocks past a warp's last query add
        # nothing to it. With 256 queries a workgroup and 64-key blocks
        # that is up to three blocks a warp, at the end of the walk: for the
        # first half of the workgroup the last two steps are wholly masked,
        # and those warps keep only the steps' barriers and their wait for
        # their share of V's last block, so the steps' math phases are the
        # other half's alone.
        skipping = causal and skip

        # ---- prologue ----------------------------------------------------
        # Q and the first three blocks in flight together: one trip to
        # memory before the first MFMA rather than three in a row.
        if paged:
            # Rows past the request's are another request's tokens: read as
            # zero, never stored.
            left = hk.max(n_q - q_row, 0)
            qf = q[hk.elem_coord(0, 0, tok0 + q_row, head * D)].load(T_Q32, bounded=left)
        else:
            qf = q[hk.elem_coord(b, head, q_row, 0)].load(T_Q32, bounded=True)
        copy_k(0, 0)
        copy_k(1, 1)
        copy_v(0, 0)
        qf *= scale_l2e
        qb_ = qf.to(bf16)
        g.wait_for(*bufs(Ks, 0))
        hk.arch.barrier()
        read_k(0)
        g.wait_for(*bufs(Ks, 1))
        end_phase()
        qk(0)
        mask(0, 0)
        m.neg_infty_()
        softmax_first(0)
        grow = wid * 0                    # no rescale pending
        hk.arch.sched_barrier()

        with hk.arch.pingpong(wid // 4 == 1):
            read_k(1)
            copy_k(2, 0)
            copy_v(1, 1)
            g.wait_for(*bufs(Vs, 0))
            end_phase()

            # ---- two steps an iteration: t = 2i (buffers 0/1), t + 1 ------
            for i in range(n_iter):
                t = 2 * i
                for s in range(2):
                    ts = t + s
                    b0, b1 = s, 1 - s
                    math_qk(ts + 1, b1, b0, grow)
                    # The LDS reads issue before the copy: they are what the
                    # next math phase waits on.
                    read_v(b0, wait=False)
                    copy_k(ts + 3, b1)
                    hk.lds.wait_for(*vrs)
                    mask(ts + 1, b1)
                    g.wait_for(*bufs(Ks, b0))
                    end_phase()
                    grow = math_pv(b1)
                    read_k(b0, wait=False)
                    copy_v(ts + 2, b0)
                    hk.lds.wait_for(*krs)
                    g.wait_for(*bufs(Vs, b1))
                    end_phase()

            # ---- the last two steps: nothing left to copy ----------------
            u = n_blk - 2

            def last_two(grow):
                math_qk(u + 1, 1, 0, grow)
                read_v(0)
                mask(u + 1, 1)
                g.wait_for(*bufs(Vs, 1))
                end_phase()
                grow = math_pv(1)
                end_phase()
                rescale(grow)
                softmax_end(1)
                end_phase()
                read_v(1)
                end_phase()
                if setprio:
                    hk.arch.setprio(1)
                pv()
                if setprio:
                    hk.arch.setprio(0)
                end_phase()

            if skipping:
                # One region or the other, each with the steps' seven
                # barriers -- a loop of 0 or 1 trips, as a branch may not
                # hold a barrier. (A branch around each phase's math held
                # the scores, P and V live into the steps, and spilled.)
                n_go = hk.min(hk.max((ctx + q_row if paged else q_row) + Q_ROWS
                                     - (blo_ + u) * KV_BLOCK, 0), 1)
                for _ in range(n_go):
                    last_two(grow)
                for _ in range(1 - n_go):
                    end_phase()
                    g.wait_for(*bufs(Vs, 1))
                    for _ in range(6):
                        end_phase()
            else:
                last_two(grow)

        # One division a query and a multiply an element: O / l divides
        # each of a lane's 64 elements in full IEEE precision, ten VALU apiece.
        inv = hk.full(T_VEC, 1.0)
        inv.__itruediv__(ell)
        O.__imul__(hk.as_vec(inv, T_O))
        if paged and kv_split:
            # A partial: normalised Oᵀ, bf16, and (m, l) in log2 units, at its
            # slot's rows; the merge (prefill_merge_kernel) weighs them.
            if slot < 0:
                o[hk.elem_coord(0, 0, tok0 + q_row, head * D)].store(
                    hk.transpose(O), bounded=left)
            else:
                prow = slot * Q_TILE + wid * Q_ROWS
                o_part[hk.elem_coord(0, 0, prow, head * D)].store(hk.transpose(O))
                ml_part[hk.elem_coord(0, head, 0, prow)] = m
                ml_part[hk.elem_coord(0, head, 1, prow)] = ell
        elif paged:
            o[hk.elem_coord(0, 0, tok0 + q_row, head * D)].store(
                hk.transpose(O), bounded=left)
        else:
            o[hk.elem_coord(b, head, q_row, 0)].store(hk.transpose(O), bounded=True)

    walk = hk.func(walk)
    if paged and kv_split:
        def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16],
                 v_cache: hk.Tensor[bf16], block_table: hk.Tensor[i32],
                 seq_lens: hk.Tensor[i32], q_start: hk.Tensor[i32],
                 tiles: hk.Tensor[i32], o: hk.Tensor[bf16], o_part: hk.Tensor[bf16],
                 ml_part: hk.Tensor[fp32]):
            walk(q, k_cache, v_cache, o, block_table, seq_lens, q_start, tiles,
                 o_part=o_part, ml_part=ml_part)

        def _grid(p):
            return (p.o.cols // D, p.tiles.rows, 1)
    elif paged:
        def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16],
                 v_cache: hk.Tensor[bf16], block_table: hk.Tensor[i32],
                 seq_lens: hk.Tensor[i32], q_start: hk.Tensor[i32],
                 tiles: hk.Tensor[i32], o: hk.Tensor[bf16]):
            walk(q, k_cache, v_cache, o, block_table, seq_lens, q_start, tiles)

        def _grid(p):
            return (p.o.cols // D, p.tiles.rows, 1)
    else:
        def body(q: hk.Tensor[bf16], k: hk.Tensor[bf16], v: hk.Tensor[bf16],
                 o: hk.Tensor[bf16]):
            if not pairs:
                walk(q, k, v, o, None, None, None, None)
                return
            # The two walks traced one after the other: in one runtime loop
            # of two trips the walk's values were held across its back edge,
            # and spilled. The second is a loop of 0 or 1 trips (the middle
            # block of an odd count has no partner), as a branch may not hold
            # the pingpong.
            lds = alloc()
            walk(q, k, v, o, None, None, None, None, 0, lds)
            for _ in range(hk.min(hk.cdiv(q.rows, Q_TILE) - 2 * hk.arch.block_idx.y - 1, 1)):
                hk.arch.barrier()
                walk(q, k, v, o, None, None, None, None, 1, lds)

        def _grid(p):
            n_t = hk.cdiv(p.q.rows, Q_TILE)
            return (p.q.depth, hk.cdiv(n_t, 2) if pairs else n_t, p.q.batch)

    body.__name__ = name
    return hk.kernel(body, arch=arch, warps=WARPS, grid=_grid, name=name, placement=placement)


def prefill_merge_kernel(name: str, *, head_dim: int = 128, arch: Optional[str] = None):
    """The KV-split paged prefill's merge: for each merge entry (first token,
    rows, first slot, splits) of `mlist` and every head, the splits'
    normalised partials weighed by l * exp2(m - max) -- a warp a (head,
    entry, 32 rows), the rows a register tile, the splits folded in turn."""
    D = head_dim
    T = hk.RT(fp32, Q_ROWS, D, "row")
    CV = hk.col_vec(T)
    chunks = Q_TILE // Q_ROWS

    def body(mlist: hk.Tensor[i32], o_part: hk.Tensor[bf16], ml_part: hk.Tensor[fp32],
             o: hk.Tensor[bf16]):
        head = hk.arch.block_idx.x
        e = hk.arch.block_idx.y // chunks
        r0 = (hk.arch.block_idx.y % chunks) * Q_ROWS
        tok0 = mlist[hk.elem_coord(0, 0, e, 0)].load_scalar(invariant=True)
        n_rows = mlist[hk.elem_coord(0, 0, e, 1)].load_scalar(invariant=True)
        slot0 = mlist[hk.elem_coord(0, 0, e, 2)].load_scalar(invariant=True)
        n_s = mlist[hk.elem_coord(0, 0, e, 3)].load_scalar(invariant=True)
        top = hk.full(CV, -1e30)
        tot = hk.zeros(CV)
        ms = hk.zeros(CV)
        ls = hk.zeros(CV)
        acc = hk.zeros(T)
        os_ = hk.zeros(T)
        for sp in range(n_s):
            ml_part[hk.elem_coord(0, head, 0, (slot0 + sp) * Q_TILE + r0)].load(CV, into=ms)
            top.max_(ms)
        for sp in range(n_s):
            prow = (slot0 + sp) * Q_TILE + r0
            ml_part[hk.elem_coord(0, head, 0, prow)].load(CV, into=ms)
            ml_part[hk.elem_coord(0, head, 1, prow)].load(CV, into=ls)
            o_part[hk.elem_coord(0, 0, prow, head * D)].load(T, into=os_)
            ms.__isub__(top)
            ms.exp2_()
            ms.__imul__(ls)
            tot.__iadd__(ms)
            os_.__imul__(ms[:, None])
            acc.__iadd__(os_)
        acc.__itruediv__(tot[:, None])
        left = hk.max(n_rows - r0, 0)
        o[hk.elem_coord(0, 0, tok0 + r0, head * D)].store(acc, bounded=left)

    body.__name__ = name
    return hk.kernel(body, arch=arch, warps=1, name=name,
                     grid=lambda p: (p.o.cols // D, p.mlist.rows * chunks, 1))


_KERNELS: Dict[Any, Any] = {}


def kernel_for(head_dim: int, causal: bool, scale: Optional[float] = None, **opts):
    """The kernel for (head_dim, causal, scale), built once. A scale other
    than 1/sqrt(head_dim) is a separate kernel, named by its bits. Causal on
    the default numbering walks query blocks in pairs (attn_kernel's
    `pairs`) unless `pairs=False`; only that is named."""
    if scale is not None and scale == head_dim ** -0.5:
        scale = None
    on = causal and opts.get("mapping", "block_first") == "block_first"
    pairs = bool(opts.pop("pairs", True)) and on
    key = (head_dim, causal, scale, pairs, tuple(sorted(opts.items())))
    if key not in _KERNELS:
        name = f"attn_fwd_d{head_dim}{'_causal' if causal else ''}_cdna"
        if scale is not None:
            import struct  # noqa: PLC0415
            name += f"_s{struct.unpack('<Q', struct.pack('<d', float(scale)))[0]:016x}"
        named = {**opts, **({"pairs": False} if on and not pairs else {})}
        for kk, vv in sorted(named.items()):
            val = int(vv) if isinstance(vv, bool) else vv
            name += "_" + "".join(ch if ch.isalnum() else "p" for ch in f"{kk}{val}")
        _KERNELS[key] = attn_kernel(name, head_dim=head_dim, causal=causal,
                                    scale=scale, pairs=pairs, **opts)
    return _KERNELS[key]


def split_kernels_for(head_dim: int, scale: Optional[float] = None):
    """The KV-split paged prefill and its merge for (head_dim, scale), built
    once (hk.ops.paged_prefill.split_plan's work)."""
    if scale is not None and scale == head_dim ** -0.5:
        scale = None
    key = ("paged_split", head_dim, scale)
    if key not in _KERNELS:
        name = f"paged_prefill_split_d{head_dim}_cdna"
        if scale is not None:
            import struct  # noqa: PLC0415
            name += f"_s{struct.unpack('<Q', struct.pack('<d', float(scale)))[0]:016x}"
        _KERNELS[key] = (
            attn_kernel(name, head_dim=head_dim, causal=True, scale=scale, paged=True,
                        kv_split=True),
            prefill_merge_kernel(f"paged_prefill_merge_d{head_dim}_cdna", head_dim=head_dim))
    return _KERNELS[key]


def paged_kernel_for(head_dim: int, scale: Optional[float] = None,
                     mapping: str = "block_first"):
    """The paged prefill (attn_kernel's `paged`) for (head_dim, scale,
    mapping), built once: hk.ops.paged_prefill's kernel on gfx950."""
    if scale is not None and scale == head_dim ** -0.5:
        scale = None
    key = ("paged", head_dim, scale, mapping)
    if key not in _KERNELS:
        name = f"paged_prefill_d{head_dim}_cdna"
        if scale is not None:
            import struct  # noqa: PLC0415
            name += f"_s{struct.unpack('<Q', struct.pack('<d', float(scale)))[0]:016x}"
        if mapping != "block_first":
            name += f"_{mapping}"
        _KERNELS[key] = attn_kernel(name, head_dim=head_dim, causal=True,
                                    scale=scale, paged=True, mapping=mapping)
    return _KERNELS[key]


__all__ = ["attn_kernel", "kernel_for", "paged_kernel_for", "split_kernels_for",
           "prefill_merge_kernel", "mapping_for", "HEAD_DIMS", "MAPPINGS",
           "Q_TILE",
           "KV_BLOCK", "PAGE"]
