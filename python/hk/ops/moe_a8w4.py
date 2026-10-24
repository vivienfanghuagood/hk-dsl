"""MoE a8w4 for gfx950 (K1): FP8 activations x AITER's MXFP4 expert weights, as vLLM's DeepSeek-V4.1
prefill stores them (AITER `shuffle_weight` / `shuffle_scale` with is_guinterleave, w13 gate|up).

Stage 1 here: h = act(gate, up) per (token, expert) pair, act = SiLU-mul with AITER's clamp
(gate <= L, |up| <= L), on gfx950's scaled MFMA with the weights as the FP4 A operand -- the
gate and up 16-row blocks are then accumulator rows, so the activation pairs subtiles in place.

A workgroup owns one expert and `nb` 16-row gate blocks (with their up blocks: 32 * nb weight rows,
16 * nb columns of h), and streams that block of the weight once per K step for all of the
expert's tokens: each warp takes `mt` 16-token tiles at a time, so a weight fragment loaded once
feeds `mt` MFMAs, and the workgroup's warps read the same fragments while they are in L2.

Inputs are in the sorted order a MoE sort makes (`prepare`): the (token, choice) pairs grouped by
expert, each expert's run padded to whole 16-row tiles; activations already MX FP8 in that order
(AITER's own stage 1 reads them so too: its quant writes them sorted).
"""

from __future__ import annotations

import functools
import math

from ..types import bf16, fp8e4m3, fp32, i8, i32
from .gemm_cdna import NUM_XCDS

TILE = 16


def stage1_kernel(name: str, *, group_rows: int, nb: int = 4, mt: int = 2, warps: int = 4,
                  limit: float = 10.0):
    """h [P, I] bf16 = act(x [P, K] fp8 (MX scales xs [P, K/32]) . w13[e]^T) for every 16-row tile of
    expert e: tiles [tiles[e], tiles[e+1]) of the sorted order (int32 [1, 1, 1, E + 1])."""
    import hk  # noqa: PLC0415

    GR = group_rows
    R = 32 * nb                                            # weight rows a workgroup: nb (gate, up) pairs
    lim = float(limit)

    def body(x: hk.Tensor[fp8e4m3], xs: hk.Tensor[i8], w: hk.Tensor[i8], ws: hk.Tensor[i8],
             tiles: hk.Tensor[i32], h: hk.Tensor[bf16]):
        blk = hk.arch.block_idx.x
        e = hk.arch.block_idx.y
        t0 = tiles[hk.elem_coord(0, 0, 0, e)].load_scalar()
        t1 = tiles[hk.elem_coord(0, 0, 0, e + 1)].load_scalar()
        wid = hk.arch.warp_id()
        row_w = e * GR + blk * (16 * nb)
        steps = x.cols // 128
        span = warps * mt                                  # tiles a round of the workgroup
        accs = [hk.zeros(hk.RT(fp32, R, TILE, "col")) for _ in range(mt)]
        for rnd in range(hk.cdiv(t1 - t0, span)):
            base = t0 + rnd * span + wid * mt
            for acc in accs:
                acc.zero_()
            for kc in range(steps):
                a = hk.load_fp4_frags(hk.RT(fp8e4m3, R, 128), w, row_w, kc, group_rows=GR,
                                      gate_up=True)
                sa = hk.load_mx_scales(ws, R, 4, row_w, kc * 4, layout="aiter_gate_up",
                                       group_rows=GR)
                for i in range(mt):
                    # a tile past the expert's last reads its neighbour's rows; its result is
                    # never stored
                    r0 = hk.min(base + i, t1 - 1) * TILE
                    b = x[hk.elem_coord(0, 0, r0, kc * 128)].load(hk.RT(fp8e4m3, TILE, 128))
                    sb = hk.load_mx_scales(xs, TILE, 4, r0, kc * 4)
                    hk.mma_ABt_mx(a, b, accs[i], sa, sb, a_fmt="fp4", into=accs[i])
            for i in range(mt):
                if base + i < t1:
                    for j in range(nb):
                        g = accs[i].subtile(TILE, 2 * j).minimum(lim)
                        u = accs[i].subtile(TILE, 2 * j + 1).maximum(-lim).minimum(lim)
                        hv = g.silu() * u
                        h[hk.elem_coord(0, 0, (base + i) * TILE, (blk * nb + j) * TILE)].store(
                            hk.transpose(hv))

    body.__name__ = name
    return hk.kernel(body, arch="gfx950", warps=warps, name=name,
                     grid=lambda p: (p.h.cols // (16 * nb), p.tiles.cols - 1, 1))


def stage1_lds_kernel(name: str, *, group_rows: int, mt: int = 4, pp: int = 2,
                      limit: float = 10.0, xcd: bool = True, exact: bool = False):
    """Stage 1, version 3 (K1 c4). Four warps; warp w owns `pp` (gate, up) pairs, so a workgroup
    covers 64 * pp columns of h and an expert takes I / (64 pp) workgroups, each reading the
    expert's token tiles once a round. A round is `mt` 16-token tiles; each 256-wide K step
    (two scaled-MFMA chunks: one AITER scale dword a lane holds both) brings the round's tiles
    into LDS once a workgroup by LDS-DMA, and a warp's FP4 weight fragments and scale dwords into
    registers, one step ahead in a second set (the loop is unrolled by two; K % 512 == 0)."""
    import hk  # noqa: PLC0415

    GR = group_rows
    warps = 4
    lim = float(limit)
    KS = 256
    R = 32 * pp
    st_x = hk.ST(fp8e4m3, TILE * mt, KS)
    t_w = hk.RT(fp8e4m3, R, KS)
    t_x = hk.RT(fp8e4m3, TILE * mt, KS)
    t_c = hk.RT(fp32, R, TILE * mt, "col")

    def body(x: hk.Tensor[fp8e4m3], xs: hk.Tensor[i8], w: hk.Tensor[i8], ws: hk.Tensor[i8],
             tiles: hk.Tensor[i32], h: hk.Tensor[bf16]):
        g = hk.group(warps)
        Xs = hk.lds.alloc(st_x, count=2, name="Xs")
        nblk = h.cols // (TILE * pp * warps)
        wg = hk.arch.block_idx.x
        if xcd:
            # gfx950 deals workgroups to its XCDs round robin (id % 8); renumber so that an
            # expert's column blocks share one XCD, and with it the L2 its token rows sit in
            total = (tiles.cols - 1) * nblk
            moved = (wg % NUM_XCDS) * (total // NUM_XCDS) + wg // NUM_XCDS
            whole = 1 - hk.min(total % NUM_XCDS, 1)        # a permutation only for whole rounds
            wg = whole * moved + (1 - whole) * wg
        blk = wg % nblk
        e = wg // nblk
        t0 = tiles[hk.elem_coord(0, 0, 0, e)].load_scalar()
        t1 = tiles[hk.elem_coord(0, 0, 0, e + 1)].load_scalar()
        wid = hk.arch.warp_id()
        col0 = (blk * warps + wid) * pp * TILE
        row_w = e * GR + col0                              # this warp's first gate block
        steps = x.cols // KS
        C = hk.zeros(t_c)
        X = hk.zeros(t_x)
        W = [hk.zeros(t_w), hk.zeros(t_w)]
        S = [hk.load_mx_scales(ws, R, KS // 32, row_w, 0, layout="aiter_gate_up", group_rows=GR)
             for _ in range(2)]
        SX = [hk.load_mx_scales(xs, TILE * mt, KS // 32, 0, 0, loose=True) for _ in range(2)]

        def fetch(kc, r0, slot):
            g.load(Xs[slot], x.at[r0, kc * KS], wait=False)
            hk.load_fp4_frags(t_w, w, row_w, kc * (KS // 128), group_rows=GR, gate_up=True,
                              into=W[slot])
            hk.load_mx_scales(ws, R, KS // 32, row_w, kc * (KS // 32), layout="aiter_gate_up",
                              group_rows=GR, into=S[slot])
            hk.load_mx_scales(xs, TILE * mt, KS // 32, r0, kc * (KS // 32), into=SX[slot], loose=True)

        def step(kc, cur, r0, prefetch=True):
            g.wait_for(Xs[cur], exact=exact)
            hk.arch.barrier()
            if prefetch:
                fetch(kc + 1, r0, 1 - cur)
            Xs[cur].load(t_x, wait=False, into=X)
            hk.lds.wait_for(X)
            hk.mma_ABt_mx(W[cur], X, C, S[cur], SX[cur], a_fmt="fp4", into=C)
            hk.arch.sched_barrier()

        for rnd in range(hk.cdiv(t1 - t0, mt)):
            r0 = (t0 + rnd * mt) * TILE
            C.zero_()
            fetch(0, r0, 0)
            for kk in range(steps // 2 - 1):
                step(2 * kk, 0, r0)
                step(2 * kk + 1, 1, r0)
            step(steps - 2, 0, r0)
            step(steps - 1, 1, r0, prefetch=False)
            hk.arch.barrier()
            for j in range(pp):
                gt = C.subtile(TILE, 2 * j).minimum(lim)
                up = C.subtile(TILE, 2 * j + 1).maximum(-lim).minimum(lim)
                hv = hk.transpose(gt.silu() * up)          # [16 * mt tokens, 16 columns]
                for i in range(mt):
                    if t0 + rnd * mt + i < t1:
                        h[hk.elem_coord(0, 0, r0 + i * TILE, col0 + j * TILE)].store(hv.subtile(TILE, i))

    body.__name__ = name
    return hk.kernel(body, arch="gfx950", warps=warps, name=name, max_vgprs=512,
                     grid=lambda p: ((p.h.cols // (TILE * pp * warps)) * (p.tiles.cols - 1), 1, 1))


def stage1_ring_kernel(name: str, *, group_rows: int, mt: int = 4, pp: int = 4, depth: int = 4,
                       limit: float = 10.0, xcd: bool = True, exact: bool = False,
                       xsplit: bool = False):
    """Stage 1, version 4 (K1 c5): stage1_lds_kernel's workgroup (warp = pp (gate, up) pairs, the
    round's mt token tiles shared through LDS), with 128-wide K steps and `depth` operand sets
    (registers for the warp's weights and scales, LDS slots for the tiles): step k+depth-1's
    operands are issued at step k, so a load has depth-1 steps of MFMAs to land under -- for the
    registers two 256-wide sets took. The loop is unrolled by `depth`; K % (128 * depth) == 0."""
    import hk  # noqa: PLC0415

    GR = group_rows
    warps = 4
    lim = float(limit)
    KS = 128
    R = 32 * pp
    st_x = hk.ST(fp8e4m3, TILE * mt, KS)
    t_w = hk.RT(fp8e4m3, R, KS)
    t_x = hk.RT(fp8e4m3, TILE * mt, KS)
    t_c = hk.RT(fp32, R, TILE * mt, "col")
    t_x1 = hk.RT(fp8e4m3, TILE, KS)
    t_c1 = hk.RT(fp32, R, TILE, "col")
    # xsplit: the round's token tiles read from LDS and multiplied one at a time, into an
    # accumulator each -- one token fragment live instead of mt (registers for bigger rounds)
    D = depth

    def body(x: hk.Tensor[fp8e4m3], xs: hk.Tensor[i8], w: hk.Tensor[i8], ws: hk.Tensor[i8],
             tiles: hk.Tensor[i32], h: hk.Tensor[bf16]):
        g = hk.group(warps)
        Xs = hk.lds.alloc(st_x, count=D, name="Xs")
        nblk = h.cols // (TILE * pp * warps)
        wg = hk.arch.block_idx.x
        if xcd:
            total = (tiles.cols - 1) * nblk
            moved = (wg % NUM_XCDS) * (total // NUM_XCDS) + wg // NUM_XCDS
            whole = 1 - hk.min(total % NUM_XCDS, 1)
            wg = whole * moved + (1 - whole) * wg
        blk = wg % nblk
        e = wg // nblk
        t0 = tiles[hk.elem_coord(0, 0, 0, e)].load_scalar()
        t1 = tiles[hk.elem_coord(0, 0, 0, e + 1)].load_scalar()
        wid = hk.arch.warp_id()
        col0 = (blk * warps + wid) * pp * TILE
        row_w = e * GR + col0
        steps = x.cols // KS
        if xsplit:
            Cs = [hk.zeros(t_c1) for _ in range(mt)]
            X1 = hk.zeros(t_x1)
        else:
            C = hk.zeros(t_c)
            X = hk.zeros(t_x)
        W = [hk.zeros(t_w) for _ in range(D)]
        S = [hk.load_mx_scales(ws, R, 4, row_w, 0, layout="aiter_gate_up", group_rows=GR)
             for _ in range(D)]
        if xsplit:
            SX = [[hk.load_mx_scales(xs, TILE, 4, 0, 0, loose=True) for _ in range(mt)]
                  for _ in range(D)]
        else:
            SX = [hk.load_mx_scales(xs, TILE * mt, 4, 0, 0, loose=True) for _ in range(D)]

        def fetch(kc, r0, slot):
            g.load(Xs[slot], x.at[r0, kc * KS], wait=False)
            hk.load_fp4_frags(t_w, w, row_w, kc, group_rows=GR, gate_up=True, into=W[slot])
            hk.load_mx_scales(ws, R, 4, row_w, kc * 4, layout="aiter_gate_up", group_rows=GR,
                              into=S[slot])
            if xsplit:
                for ti in range(mt):
                    hk.load_mx_scales(xs, TILE, 4, r0 + ti * TILE, kc * 4, into=SX[slot][ti],
                                      loose=True)
            else:
                hk.load_mx_scales(xs, TILE * mt, 4, r0, kc * 4, into=SX[slot], loose=True)

        def step(kc, cur, r0, prefetch=True):
            g.wait_for(Xs[cur], exact=exact)
            hk.arch.barrier()
            if prefetch:
                fetch(kc + D - 1, r0, (cur + D - 1) % D)
            if xsplit:
                for ti in range(mt):
                    Xs[cur].subtile(TILE, KS, ti, 0).load(t_x1, wait=False, into=X1)
                    hk.lds.wait_for(X1)
                    hk.mma_ABt_mx(W[cur], X1, Cs[ti], S[cur], SX[cur][ti], a_fmt="fp4", into=Cs[ti])
            else:
                Xs[cur].load(t_x, wait=False, into=X)
                hk.lds.wait_for(X)
                hk.mma_ABt_mx(W[cur], X, C, S[cur], SX[cur], a_fmt="fp4", into=C)
            hk.arch.sched_barrier()

        for rnd in range(hk.cdiv(t1 - t0, mt)):
            r0 = (t0 + rnd * mt) * TILE
            if xsplit:
                for c_ in Cs:
                    c_.zero_()
            else:
                C.zero_()
            for k in range(D - 1):
                fetch(k, r0, k)
            for kk in range(steps // D - 1):
                for u in range(D):
                    step(kk * D + u, u, r0)
            last = steps - D
            for u in range(D):
                step(last + u, u, r0, prefetch=u == 0)
            hk.arch.barrier()
            if xsplit:
                for i in range(mt):
                    if t0 + rnd * mt + i < t1:
                        for j in range(pp):
                            gt = Cs[i].subtile(TILE, 2 * j).minimum(lim)
                            up = Cs[i].subtile(TILE, 2 * j + 1).maximum(-lim).minimum(lim)
                            h[hk.elem_coord(0, 0, r0 + i * TILE, col0 + j * TILE)].store(
                                hk.transpose(gt.silu() * up))
            else:
                for j in range(pp):
                    gt = C.subtile(TILE, 2 * j).minimum(lim)
                    up = C.subtile(TILE, 2 * j + 1).maximum(-lim).minimum(lim)
                    hv = hk.transpose(gt.silu() * up)
                    for i in range(mt):
                        if t0 + rnd * mt + i < t1:
                            h[hk.elem_coord(0, 0, r0 + i * TILE, col0 + j * TILE)].store(hv.subtile(TILE, i))

    body.__name__ = name
    return hk.kernel(body, arch="gfx950", warps=warps, name=name, max_vgprs=512,
                     grid=lambda p: ((p.h.cols // (TILE * pp * warps)) * (p.tiles.cols - 1), 1, 1))


def stage1_dma_kernel(name: str, *, group_rows: int, mt: int = 4, pp: int = 4, depth: int = 4,
                      limit: float = 10.0):
    """Stage 1, version 5 (K1 c6): the weights through LDS too. Every in-flight load of the K loop
    is then an LDS-DMA copy -- the weights' 1 KB chunks (AITER's layout viewed [E * N/16 * 2,
    K/2 * 16]: a workgroup's rows are contiguous) and the round's token tiles -- issued depth-1
    128-wide steps ahead, with exact waits. Only the two scale sets are register loads, one step
    ahead and issued before the step's copies, so the copy waits that also retire them retire
    nothing early. Loop unrolled by `depth` (even); K % (128 * depth) == 0."""
    import hk  # noqa: PLC0415

    GR = group_rows
    warps = 4
    lim = float(limit)
    KS = 128
    R = 32 * pp
    RW = 2 * pp * warps                                     # weight rows (16-row block halves) a workgroup
    st_x = hk.ST(fp8e4m3, TILE * mt, KS)
    st_w = hk.ST(fp8e4m3, RW, 1024)
    t_w = hk.RT(fp8e4m3, R, KS)
    t_x = hk.RT(fp8e4m3, TILE * mt, KS)
    t_c = hk.RT(fp32, R, TILE * mt, "col")
    D = depth
    assert D % 2 == 0 and pp >= 2, "an even depth; pp >= 2 (16-row LDS weight tiles)"

    def body(x: hk.Tensor[fp8e4m3], xs: hk.Tensor[i8], wv: hk.Tensor[fp8e4m3], ws: hk.Tensor[i8],
             tiles: hk.Tensor[i32], h: hk.Tensor[bf16]):
        g = hk.group(warps)
        Xs = hk.lds.alloc(st_x, count=D, name="Xs")
        Ws = hk.lds.alloc(st_w, count=D, name="Ws", shape="16x128")
        nblk = h.cols // (TILE * pp * warps)
        wg = hk.arch.block_idx.x
        total = (tiles.cols - 1) * nblk
        moved = (wg % NUM_XCDS) * (total // NUM_XCDS) + wg // NUM_XCDS
        whole = 1 - hk.min(total % NUM_XCDS, 1)
        wg = whole * moved + (1 - whole) * wg
        blk = wg % nblk
        e = wg // nblk
        t0 = tiles[hk.elem_coord(0, 0, 0, e)].load_scalar()
        t1 = tiles[hk.elem_coord(0, 0, 0, e + 1)].load_scalar()
        wid = hk.arch.warp_id()
        col0 = (blk * warps + wid) * pp * TILE
        row_w = e * GR + col0                              # the warp's first gate row (scales)
        wrow0 = e * (GR // TILE) + blk * RW                # the workgroup's first weight-view row
        steps = x.cols // KS
        C = hk.zeros(t_c)
        X = hk.zeros(t_x)
        W = hk.zeros(t_w)
        S = [hk.load_mx_scales(ws, R, 4, row_w, 0, layout="aiter_gate_up", group_rows=GR)
             for _ in range(2)]
        SX = [hk.load_mx_scales(xs, TILE * mt, 4, 0, 0, loose=True) for _ in range(2)]

        def scales(kc, r0, q):
            hk.load_mx_scales(ws, R, 4, row_w, kc * 4, layout="aiter_gate_up", group_rows=GR,
                              into=S[q])
            hk.load_mx_scales(xs, TILE * mt, 4, r0, kc * 4, into=SX[q], loose=True)

        def copies(kc, r0, slot):
            g.load(Ws[slot], wv.at[wrow0, kc * 1024], wait=False)
            g.load(Xs[slot], x.at[r0, kc * KS], wait=False)

        def step(kc, u, r0, more_scales=True, more_copies=True):
            g.wait_for(Ws[u], Xs[u])
            hk.arch.barrier()
            if more_scales:
                scales(kc + 1, r0, (u + 1) % 2)
            if more_copies:
                copies(kc + D - 1, r0, (u + D - 1) % D)
            hk.load_fp4_frags_lds(t_w, Ws[u], wid * 2 * pp, 0, into=W)
            Xs[u].load(t_x, wait=False, into=X)
            hk.lds.wait_for(X)
            hk.mma_ABt_mx(W, X, C, S[u % 2], SX[u % 2], a_fmt="fp4", into=C)
            hk.arch.sched_barrier()

        for rnd in range(hk.cdiv(t1 - t0, mt)):
            r0 = (t0 + rnd * mt) * TILE
            C.zero_()
            scales(0, r0, 0)
            for k in range(D - 1):
                copies(k, r0, k)
            for kk in range(steps // D - 1):
                for u in range(D):
                    step(kk * D + u, u, r0)
            last = steps - D
            for u in range(D):
                step(last + u, u, r0, more_scales=u < D - 1, more_copies=u == 0)
            hk.arch.barrier()
            for j in range(pp):
                gt = C.subtile(TILE, 2 * j).minimum(lim)
                up = C.subtile(TILE, 2 * j + 1).maximum(-lim).minimum(lim)
                hv = hk.transpose(gt.silu() * up)
                for i in range(mt):
                    if t0 + rnd * mt + i < t1:
                        h[hk.elem_coord(0, 0, r0 + i * TILE, col0 + j * TILE)].store(hv.subtile(TILE, i))

    body.__name__ = name
    return hk.kernel(body, arch="gfx950", warps=warps, name=name, max_vgprs=512,
                     grid=lambda p: ((p.h.cols // (TILE * pp * warps)) * (p.tiles.cols - 1), 1, 1))


@functools.lru_cache(maxsize=None)
def _stage1_dma(group_rows: int, mt: int, pp: int, depth: int, limit: float):
    return stage1_dma_kernel(f"moe_a8w4_s1d_gr{group_rows}_mt{mt}_pp{pp}_d{depth}",
                             group_rows=group_rows, mt=mt, pp=pp, depth=depth, limit=limit)


def weight_view(w13, group_rows: int):
    """AITER's shuffled w13 bytes as stage1_dma_kernel reads them: fp8-typed [E * GR/16, K/2 * 16],
    a row a (expert, 16-row block, gate|up) and 1 KB a 128-wide K step (a view, no copy)."""
    import torch  # noqa: PLC0415

    b = w13.view(torch.uint8)
    rows = b.shape[0] // 16
    return b.reshape(rows, -1).view(torch.float8_e4m3fn)


def stage1_direct_kernel(name: str, *, group_rows: int, k: int, mt: int = 4, pp: int = 4,
                         depth: int = 3, limit: float = 10.0):
    """Stage 1, version 6 (K1 c7): no LDS. Every operand is a register load -- the warp's FP4 weight
    fragments, its scale dwords, and the round's token fragments (all four warps read the same
    ones; L2 serves the repeats) -- so loads retire in order on vmcnt and the compiler's own waits
    are exact (no copy wait retiring later loads, c5). The K loop is unrolled at trace time (k is
    the hidden size, a build constant) over a ring of `depth` operand sets, step s+depth-1 issued
    before step s's MFMAs: the schedule AITER's FlyDSL stage 1 has (no loop, loads far ahead)."""
    import hk  # noqa: PLC0415

    GR = group_rows
    warps = 4
    lim = float(limit)
    KS = 128
    R = 32 * pp
    steps = k // KS
    assert k % KS == 0 and steps >= depth
    t_w = hk.RT(fp8e4m3, R, KS)
    t_x = hk.RT(fp8e4m3, TILE * mt, KS)
    t_c = hk.RT(fp32, R, TILE * mt, "col")
    D = depth

    def body(x: hk.Tensor[fp8e4m3], xs: hk.Tensor[i8], w: hk.Tensor[i8], ws: hk.Tensor[i8],
             tiles: hk.Tensor[i32], units: hk.Tensor[i32], h: hk.Tensor[bf16]):
        # one round a workgroup (AITER's shape: no loop for the unrolled K steps' addresses to be
        # hoisted out of): units[u] = (expert, first tile) of round u, from units_for()
        nblk = h.cols // (TILE * pp * warps)
        wg = hk.arch.block_idx.x
        blk = wg % nblk
        u = wg // nblk
        e = units[hk.elem_coord(0, 0, u, 0)].load_scalar()
        tb = units[hk.elem_coord(0, 0, u, 1)].load_scalar()
        t1 = tiles[hk.elem_coord(0, 0, 0, e + 1)].load_scalar()
        t0 = tb
        wid = hk.arch.warp_id()
        col0 = (blk * warps + wid) * pp * TILE
        row_w = e * GR + col0
        C = hk.zeros(t_c)

        def fetch(kc, r0, q, W, X, S, SX):
            # each set is made by its first load (no zero fill: a tile only loads write and MFMAs
            # read may live in AGPRs, as AITER keeps its weights), loaded into afterwards
            W[q] = hk.load_fp4_frags(t_w, w, row_w, kc, group_rows=GR, gate_up=True, into=W[q])
            S[q] = hk.load_mx_scales(ws, R, 4, row_w, kc * 4, layout="aiter_gate_up",
                                     group_rows=GR, into=S[q])
            X[q] = x[hk.elem_coord(0, 0, r0, kc * KS)].load(t_x, into=X[q])
            SX[q] = hk.load_mx_scales(xs, TILE * mt, 4, r0, kc * 4, into=SX[q], loose=True)

        for rnd in range(1):
            r0 = t0 * TILE
            W, X, S, SX = [None] * D, [None] * D, [None] * D, [None] * D
            for s in range(D - 1):
                fetch(s, r0, s, W, X, S, SX)
            for s in range(steps):
                if s + D - 1 < steps:
                    fetch(s + D - 1, r0, (s + D - 1) % D, W, X, S, SX)
                q = s % D
                hk.mma_ABt_mx(W[q], X[q], C, S[q], SX[q], a_fmt="fp4", into=C)
                # keep each step's loads where the ring puts them: hoisted across the unrolled
                # steps they hold every set at once (spills)
                hk.arch.sched_barrier()
            for j in range(pp):
                gt = C.subtile(TILE, 2 * j).minimum(lim)
                up = C.subtile(TILE, 2 * j + 1).maximum(-lim).minimum(lim)
                hv = hk.transpose(gt.silu() * up)
                for i in range(mt):
                    if t0 + i < t1:
                        h[hk.elem_coord(0, 0, r0 + i * TILE, col0 + j * TILE)].store(hv.subtile(TILE, i))

    body.__name__ = name
    return hk.kernel(body, arch="gfx950", warps=warps, name=name, max_vgprs=512,
                     grid=lambda p: ((p.h.cols // (TILE * pp * warps)) * p.units.rows, 1, 1))


def units_for(tiles, mt: int):
    """[n, 2] int32 (expert, first tile) for every round of `mt` 16-row tiles of every expert."""
    import torch  # noqa: PLC0415

    t = tiles.view(-1).long()
    n = (t[1:] - t[:-1] + mt - 1) // mt
    e = torch.repeat_interleave(torch.arange(n.numel(), device=t.device), n)
    first = torch.repeat_interleave(t[:-1], n)
    start = torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
    r = torch.arange(e.numel(), device=t.device) - start
    return torch.stack([e, first + r * mt], 1).to(torch.int32).view(1, 1, -1, 2)


def ring_round(rows: int, n_experts: int, k: int):
    """(mt, depth) of the xsplit ring kernel for P sorted rows over E experts (host values, no sync).
    An expert's average t tiles are cut into the fewest rounds of at most 7 tiles, as even as
    possible; a round is never under 2 tiles. Depth 2 (K % 256 == 0; deeper rings measured no
    better). Measured on MI355X, Kimi K3 shapes (pp 3), uniform routing, M 640..16384 (K1 c9):
    one round an expert where it fits beats smaller rounds by up to 1.2x."""
    if k % 256:
        raise ValueError(f"moe_a8w4.ring_round: K={k}, a multiple of 256")
    t = rows / TILE / max(n_experts, 1)
    r = max(1, math.ceil(t / 7))
    return min(7, max(2, math.ceil(t / r))), 2


@functools.lru_cache(maxsize=None)
def _stage1_direct(group_rows: int, k: int, mt: int, pp: int, depth: int, limit: float):
    return stage1_direct_kernel(f"moe_a8w4_s1x_gr{group_rows}_k{k}_mt{mt}_pp{pp}_d{depth}",
                                group_rows=group_rows, k=k, mt=mt, pp=pp, depth=depth, limit=limit)


@functools.lru_cache(maxsize=None)
def _stage1_ring(group_rows: int, mt: int, pp: int, depth: int, limit: float, exact: bool = False,
                 xsplit: bool = False):
    return stage1_ring_kernel(f"moe_a8w4_s1r_gr{group_rows}_mt{mt}_pp{pp}_d{depth}_e{int(exact)}"
                              f"_x{int(xsplit)}", group_rows=group_rows, mt=mt, pp=pp, depth=depth,
                              limit=limit, exact=exact, xsplit=xsplit)


@functools.lru_cache(maxsize=None)
def _stage1(group_rows: int, nb: int, mt: int, warps: int, limit: float):
    return stage1_kernel(f"moe_a8w4_s1_gr{group_rows}_nb{nb}_mt{mt}_w{warps}", group_rows=group_rows,
                         nb=nb, mt=mt, warps=warps, limit=limit)


def prepare(hidden, topk_ids, n_experts: int):
    """The sort and MX FP8 quant a stage 1 reads (torch; AITER fuses this into one kernel):
    (x fp8 [P, K], xs uint8 [P, K/32], tiles int32 [1, 1, 1, E + 1], pairs int32 [P] (token * topk +
    choice, -1 padding)). P = the experts' pair counts, each padded to whole 16-row tiles."""
    import torch  # noqa: PLC0415

    m, topk = topk_ids.shape
    flat = topk_ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=n_experts)
    ntile = (counts + TILE - 1) // TILE
    tiles = torch.zeros(n_experts + 1, dtype=torch.int64, device=hidden.device)
    tiles[1:] = torch.cumsum(ntile, 0)
    p = int(tiles[-1]) * TILE
    start = torch.zeros_like(counts)
    start[1:] = torch.cumsum(counts, 0)[:-1]
    dst = tiles[:-1][flat[order]] * TILE + (torch.arange(order.numel(), device=hidden.device)
                                            - start[flat[order]])
    pairs = torch.full((p,), -1, dtype=torch.int64, device=hidden.device)
    pairs[dst] = order
    rows = hidden.new_zeros((p, hidden.shape[1]))
    live = pairs >= 0
    rows[live] = hidden[pairs[live] // topk]
    xq, xs = mx_quant(rows)
    return xq, xs, tiles.to(torch.int32).view(1, 1, 1, -1), pairs.to(torch.int32)


def aiter_layout_w13(w):
    """[E, 2I, K/2] uint8 (gate rows, then up) -> AITER's gfx950 MoE layout (shuffle_weight,
    is_guinterleave, gate_up), as a [E * 2I, K/2] uint8 view. For tests: vLLM's weights are stored so."""
    e, n, kp = w.shape
    x = w.view(e, 2, n // 32, 16, kp // 64, 4, 16)
    return x.permute(0, 2, 1, 4, 5, 3, 6).contiguous().view(e * n, kp)


def aiter_layout_w13_scale(s, e: int):
    """[E * 2I, K/32] uint8 E8M0 -> AITER's shuffle_scale(is_guinterleave, gate_up) layout."""
    rows, kb = s.shape
    x = s.view(e, 2, rows // e // 32, 16, kb // 8, 2, 4)
    return x.permute(0, 2, 4, 6, 3, 5, 1).contiguous().view(rows, kb)


def mx_quant(x):
    """Rows of x to FP8 e4m3 with an E8M0 scale per 32 (OCP MX: scale 2^(floor(log2 amax) - 8))."""
    import torch  # noqa: PLC0415

    m, k = x.shape
    v = x.float().view(m, k // 32, 32)
    amax = v.abs().amax(-1).clamp_min(2.0 ** -126)
    e = torch.floor(torch.log2(amax)) - 8
    q = (v / torch.pow(2.0, e)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(m, k), (e + 127).clamp(0, 254).to(torch.uint8)


@functools.lru_cache(maxsize=None)
def _stage1_lds(group_rows: int, mt: int, limit: float, pp: int, xcd: bool = True,
                exact: bool = False):
    return stage1_lds_kernel(f"moe_a8w4_s1l_gr{group_rows}_mt{mt}_pp{pp}_x{int(xcd)}_e{int(exact)}",
                             group_rows=group_rows, mt=mt, pp=pp, limit=limit, xcd=xcd, exact=exact)


def stage1(xq, xs, w13, w13_scale, tiles, *, inter: int, nb: int = 4, mt: int = 2, warps: int = 4,
           limit: float = 10.0, lds: bool = False, pp: int = 2, xcd: bool = True, depth: int = 0,
           dma: bool = False, direct: bool = False, exact: bool = False, xsplit: bool = False,
           auto: bool = False):
    """h [P, inter] bf16 for prepare()'s sorted rows. w13 / w13_scale: AITER's shuffled uint8 views
    [E * 2 * inter, K/2] and [E * 2 * inter, K/32]. auto: the xsplit ring kernel (exact waits, the
    widest pp of 4, 3, 2, 1 whose 64 * pp columns divide inter) with ring_round()'s round size; mt, depth, pp and the mode flags are then
    ignored."""
    import torch  # noqa: PLC0415

    if auto:
        mt, depth = ring_round(xq.shape[0], tiles.numel() - 1, xq.shape[1])
        pp = next(p for p in (4, 3, 2, 1) if inter % (TILE * p * 4) == 0)
        exact, xsplit, dma, direct = True, True, False, False
    cols = 16 * nb if not (lds or depth or dma or direct) else TILE * pp * 4
    if inter % cols:
        # the grids are inter // cols column blocks: the rest would never be written
        raise ValueError(f"moe_a8w4.stage1: inter={inter} not a multiple of the {cols} columns "
                         f"a workgroup writes (pp={pp}, nb={nb})")
    # the K loops are unrolled by depth (ring, dma) or two 256-wide steps (lds), with no remainder
    unroll = 0 if direct or not (lds or depth or dma) else 128 * (depth or 4) if (dma or depth) else 512
    if unroll and xq.shape[1] % unroll or dma and (depth or 4) % 2:
        raise ValueError(f"moe_a8w4.stage1: K={xq.shape[1]} not a multiple of the {unroll} the K loop "
                         f"is unrolled by (depth={depth}{', even for dma' if dma else ''})")

    h = torch.empty((xq.shape[0], inter), device=xq.device, dtype=torch.bfloat16)
    if direct:
        _stage1_direct(2 * inter, xq.shape[1], mt, pp, depth or 3, float(limit))(
            xq, xs.view(torch.int8), w13.view(torch.int8), w13_scale.view(torch.int8), tiles,
            units_for(tiles, mt), h)
        return h
    if dma:
        _stage1_dma(2 * inter, mt, pp, depth or 4, float(limit))(
            xq, xs.view(torch.int8), weight_view(w13, 2 * inter), w13_scale.view(torch.int8), tiles, h)
        return h
    if depth:
        k = _stage1_ring(2 * inter, mt, pp, depth, float(limit), exact, xsplit)
    else:
        k = _stage1_lds(2 * inter, mt, float(limit), pp, xcd, exact) if lds else _stage1(2 * inter, nb, mt, warps, float(limit))
    k(xq, xs.view(torch.int8), w13.view(torch.int8), w13_scale.view(torch.int8), tiles, h)
    return h
