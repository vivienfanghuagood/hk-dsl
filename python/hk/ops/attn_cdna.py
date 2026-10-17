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
    warps (`skip`).
  * **Any N.** Q rows past N read zeros and are not stored; a KV block that
    runs past N_kv is masked with -inf (the DMA reads zeros there); an odd
    block count gets one fully masked block, so the walk is whole steps.
  * **(B, H, N, D)**, contiguous rows of one head, where HipKittens takes
    (B, N, H, D).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .. import dsl as hk
from ..types import bf16, fp32

NEG_INF = float("-inf")

#: Query rows a warp owns, warps a workgroup, query rows a workgroup.
Q_ROWS, WARPS = 32, 8
Q_TILE = Q_ROWS * WARPS
#: The KV block.
KV_BLOCK = 64
#: Head dims with a schedule.
HEAD_DIMS = (64, 128)


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
                skip: bool = True, _nomask: bool = False,
                arch: Optional[str] = None):
    """`lazy`: the running max is taken as stale until a block's max exceeds
    it by more than this (log2 units), so P stays below 2^lazy and O is
    rescaled only when the max really moves (FlashAttention 4's threshold,
    HipKittens' 8). 0 rescales whenever it grows at all.

    `sched` keeps HipKittens' interleave of MFMAs with VALU and exp2 in the
    math phases; `setprio` its s_setprio around the PV phase; `lpt` reverses
    the query blocks under `causal`, so the longest workgroups start first;
    `skip` drops, under `causal`, the last two steps' math for the warps
    whose queries they are wholly past. `_nomask` (wrong answers) is for
    timing what the masks cost.
    """
    D = head_dim
    if D not in HEAD_DIMS:
        raise ValueError(f"head_dim={D}: have {HEAD_DIMS}")
    scale_l2e = (scale if scale is not None else D ** -0.5) * math.log2(math.e)
    half = KV_BLOCK // 2
    # MFMAs a phase issues: QK is (64 x D) . (D x 32) in 32x32x16 steps, PV
    # (D x 64) . (64 x 32); both are 2 * D / 16.
    n_mma = 2 * D // 16

    T_Q32 = hk.RT(fp32, Q_ROWS, D, "row")      # this warp's Q, scaled in fp32
    T_K = hk.RT(bf16, KV_BLOCK, D, "row")      # a K block, [kv, d]
    T_V = hk.RT(bf16, KV_BLOCK, D, "col")      # a V block, [kv, d]
    T_S = hk.RT(fp32, KV_BLOCK, Q_ROWS, "col", frag=(32, 32))     # Sᵀ, [kv, q]
    T_P = hk.RT(bf16, KV_BLOCK, Q_ROWS, "col", frag=(16, 32, 4))  # Pᵀ as an operand
    T_O = hk.RT(fp32, D, Q_ROWS, "col", frag=(32, 32))            # Oᵀ, [d, q]
    T_VEC = hk.row_vec(T_S)                    # one value per query
    ST = hk.ST(bf16, KV_BLOCK, D)

    def body(q: hk.Tensor[bf16], k: hk.Tensor[bf16], v: hk.Tensor[bf16],
             o: hk.Tensor[bf16]):
        g = hk.group(WARPS)
        Ks = hk.lds.alloc(ST, count=2, name="Ks")
        Vs = hk.lds.alloc(ST, count=2, name="Vs")

        n_q, n_kv = q.rows, k.rows
        b = hk.arch.block_idx.z
        x = hk.arch.block_idx.x
        # GQA: workgroup x of a query block takes KV head x % H_KV, so
        # consecutive workgroups -- dealt round robin to the XCDs -- read
        # different KV heads, and the query heads that share one KV head land
        # on one XCD and share its L2 (HipKittens' numbering).
        h_kv = k.depth
        group = q.depth // h_kv
        head_kv = x % h_kv
        head = head_kv * group + x // h_kv
        qb = hk.arch.block_idx.y
        if causal and lpt:
            qb = hk.cdiv(n_q, Q_TILE) - 1 - qb
        q0 = qb * Q_TILE
        wid = hk.arch.warp_id()
        q_row = q0 + wid * Q_ROWS

        # KV blocks this workgroup walks: all of them, or under causal up to
        # its last query; then a whole number of steps of two, and at least
        # four (the blocks past N_kv are masked), so the loop below runs at
        # least once and the compiler can see it: with a path around the
        # loop, values for the epilogue were held across it, and spilled.
        n_blk = hk.cdiv(n_kv, KV_BLOCK)
        if causal:
            n_blk = hk.min(n_blk, hk.cdiv(hk.min(q0 + Q_TILE, n_q), KV_BLOCK))
        n_iter = hk.max(hk.cdiv(n_blk, 2) - 1, 1)
        n_blk = 2 * n_iter + 2

        def copy_k(t, par):
            g.load(Ks[par], k[hk.elem_coord(b, head_kv, t * KV_BLOCK, 0)], wait=False)

        def copy_v(t, par):
            g.load(Vs[par], v[hk.elem_coord(b, head_kv, t * KV_BLOCK, 0)], wait=False)

        kr = hk.zeros(T_K)
        vr = hk.zeros(T_V)
        # Two blocks of scores in flight, each one tile -- one reduction and
        # one conversion over 64 rows -- whose 32-row halves (subtiles, the
        # same registers) take their exp2 in different phases.
        S = [hk.zeros(T_S) for _ in range(2)]
        Sh = [[S[bb].subtile(half, h) for h in range(2)] for bb in range(2)]
        P_op = hk.zeros(T_P)
        O = hk.zeros(T_O)
        m = hk.zeros(T_VEC)          # the running max S is taken against
        # The new running max, and once it is taken, exp2(m_old - m): the
        # rescale O takes at the top of the next math phase.
        m_new = hk.full(T_VEC, 1.0)
        sc = m_new
        ell = hk.zeros(T_VEC)

        def read_k(par, wait=True):
            Ks[par].load(T_K, wait=False, into=kr)
            if wait:
                hk.lds.wait_for(kr)

        def read_v(par, wait=True):
            Vs[par].load(T_V, wait=False, into=vr)
            if wait:
                hk.lds.wait_for(vr)

        def qk(bb):
            S[bb].zero_()
            S[bb] += kr @ qb_.T

        def mask(t, bb):
            # Sᵀ rows are KV positions and columns queries, so the causal
            # mask (kv > q) is triu and the KV tail rows a lower fill. Each
            # behind a wave-uniform branch: most blocks need neither.
            kv0 = t * KV_BLOCK
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
            O.__iadd__(vr.T @ P_op)

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
        qf = q[hk.elem_coord(b, head, q_row, 0)].load(T_Q32, bounded=True)
        copy_k(0, 0)
        copy_k(1, 1)
        copy_v(0, 0)
        qf *= scale_l2e
        qb_ = qf.to(bf16)
        g.wait_for(Ks[0])
        hk.arch.barrier()
        read_k(0)
        g.wait_for(Ks[1])
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
            g.wait_for(Vs[0])
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
                    hk.lds.wait_for(vr)
                    mask(ts + 1, b1)
                    g.wait_for(Ks[b0])
                    end_phase()
                    grow = math_pv(b1)
                    read_k(b0, wait=False)
                    copy_v(ts + 2, b0)
                    hk.lds.wait_for(kr)
                    g.wait_for(Vs[b1])
                    end_phase()

            # ---- the last two steps: nothing left to copy ----------------
            u = n_blk - 2

            def last_two(grow):
                math_qk(u + 1, 1, 0, grow)
                read_v(0)
                mask(u + 1, 1)
                g.wait_for(Vs[1])
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
                n_go = hk.min(hk.max(q_row + Q_ROWS - u * KV_BLOCK, 0), 1)
                for _ in range(n_go):
                    last_two(grow)
                for _ in range(1 - n_go):
                    end_phase()
                    g.wait_for(Vs[1])
                    for _ in range(6):
                        end_phase()
            else:
                last_two(grow)

        # One division a query and a multiply an element: O / l divides
        # each of a lane's 64 elements in full IEEE precision, ten VALU apiece.
        inv = hk.full(T_VEC, 1.0)
        inv.__itruediv__(ell)
        O.__imul__(hk.as_vec(inv, T_O))
        o[hk.elem_coord(b, head, q_row, 0)].store(hk.transpose(O), bounded=True)

    body.__name__ = name

    def _grid(p):
        return (p.q.depth, hk.cdiv(p.q.rows, Q_TILE), p.q.batch)

    return hk.kernel(body, arch=arch, warps=WARPS, grid=_grid, name=name)


_KERNELS: Dict[Any, Any] = {}


def kernel_for(head_dim: int, causal: bool, scale: Optional[float] = None, **opts):
    """The kernel for (head_dim, causal, scale), built once. A scale other
    than 1/sqrt(head_dim) is a separate kernel, named by its bits."""
    if scale is not None and scale == head_dim ** -0.5:
        scale = None
    key = (head_dim, causal, scale, tuple(sorted(opts.items())))
    if key not in _KERNELS:
        name = f"attn_fwd_d{head_dim}{'_causal' if causal else ''}_cdna"
        if scale is not None:
            import struct  # noqa: PLC0415
            name += f"_s{struct.unpack('<Q', struct.pack('<d', float(scale)))[0]:016x}"
        for kk, vv in sorted(opts.items()):
            val = int(vv) if isinstance(vv, bool) else vv
            name += "_" + "".join(ch if ch.isalnum() else "p" for ch in f"{kk}{val}")
        _KERNELS[key] = attn_kernel(name, head_dim=head_dim, causal=causal,
                                    scale=scale, **opts)
    return _KERNELS[key]


__all__ = ["attn_kernel", "kernel_for", "HEAD_DIMS", "Q_TILE", "KV_BLOCK"]
