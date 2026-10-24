"""bf16 GEMM with fp32 accumulate on gfx950 (CDNA4, MI355X), in the DSL.

C = A·Bᵀ, A (M, K) and B (N, K), as hk.ops.gemm on RDNA. The schedule is
HipKittens' CDNA4 GEMM (kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp)
written in hk, and it is what the three gfx950 features were added for:

  * **The DMA, left in flight.** `g.load(..., wait=False)` is
    buffer_load_dwordx4 ... lds: 16 bytes a lane straight into LDS, counted on
    vmcnt. `g.wait_for(tile)` is the vmcnt, derived by hk-lds-pipeline from the
    order the copies were issued; HipKittens writes `s_waitcnt vmcnt(6)` by
    hand. Each tile is its own named buffer, so the pass knows which copy a
    read waits on and which reads a copy overwrites.
  * **Eight waves in two halves, a barrier apart** (`hk.arch.pingpong`). Each
    phase between two barriers is either memory (LDS reads, one copy) or math
    (sixteen MFMAs at raised priority), and the half of the workgroup with
    warp row 1 runs one barrier behind: on every SIMD one wave is at the
    matrix core while the other is at memory.
  * **Reads that cross a barrier.** A warp's next B fragment is read in one
    memory phase and used two phases later. The pass lets a read of a named
    buffer stay in flight across a barrier, and checks instead that no copy
    overwrites the buffer until every warp's reads of it have landed, two
    barriers back inside the pingpong.

Translating HipKittens' schedule found one thing its hand-counted waits do not
cover: the fourth memory phase reads Bs[1][0] before any vmcnt has retired
the copy into it (the vmcnt(6) a phase earlier leaves exactly that copy
outstanding), so it is right by timing. The pass refused it. Here that read
moves one phase later, into the fifth memory phase with A's half
(`late_b`), after the fourth phase's wait, and HipKittens' own counts then
cover every read; waiting for the copy where it was read instead
(vmcnt(4)) leaves fewer copies in flight and is 0.3-0.6% slower.

The block is block_m x block_n in four quadrants: A and B are each staged as
two halves, and every warp of the warps_m x warps_n grid owns one
(half_m / warps_m) x (half_n / warps_n) accumulator in each quadrant. Each
K-tile is four memory/math phase pairs, one per quadrant, and the loop body
two K-tiles, so the LDS buffer of a tile is a Python int.

Where the time goes, measured on an MI355X (docs/history/33-cdna-gemm.md):
the loop matches HipKittens' instruction for instruction and runs within 1%
of it; with C's store switched off (the accumulators kept live) the kernel
is level with hipBLASLt at 8192³. The store is 5-8% of the kernel: one
workgroup of 128 KB of LDS fills a CU, so every CU writes its tile at once,
nothing overlaps it, and its cost follows its bytes. Storing the first two
quadrants as soon as they are final (`early_store`) takes 0.4% of it back.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .. import dsl as hk
from ..types import bf16, fp32, i32
from . import _epilogue

#: An MI355X has eight XCDs; workgroups are dealt to them round robin.
NUM_XCDS = 8


def gemm_kernel(name: str, *, block_m: int = 256, block_n: int = 256,
                k_step: int = 64, warps_m: int = 2, warps_n: int = 4,
                wgm: int = 8, xcd_chunk: int = 64, setprio: bool = True,
                transposed_c: bool = False, early_store: bool = True,
                hipkittens_waits: bool = False, late_b: bool = True,
                splits: int = 1, lds_shape: Optional[str] = None, nt: str = "",
                stream_k: int = 0, sk_grid: int = 256,
                arch: Optional[str] = None,
                max_vgprs: int = 256, min_occupancy: int = 1, epilogue=None):
    """HipKittens' tiling by default: 256x256x64, eight warps as 2 x 4.

    `epilogue` (hk.ops._epilogue) replaces C's store with hk.templates.gemm's
    hook; a split kernel (splits > 1) stores its fp32 shares as ever and the
    hook goes on the reduction.

    `stream_k` makes it persistent: `sk_grid` workgroups (one a CU), each
    running the segments -- a block and a range of its K -- that sk_plan
    gives it (SK_MODES: 1 cuts every block's K into equal runs, stream-K
    proper; 2 deals whole blocks round robin for the grid's whole waves and
    cuts only the rest). A grid of fewer blocks than CUs, or a part wave,
    keeps every CU busy; a block whose K is cut is summed in fp32 by the
    last of its workgroups to finish, in a fixed order, and only then stored
    (epilogue included). Takes (a, b, c, ws, cnt, plan) -- sk_scratch's and
    sk_plan's -- and K a multiple of 2 * k_step.

    `hipkittens_waits` puts HipKittens' own vmcnts back (all but three copies
    after the prologue and at the end of each iteration). hk-lds-pipeline
    refuses them -- the fourth memory phase reads Bs[2] with its copy still in
    flight -- which tests/hk/ir/test_cdna_gemm.py holds.
    """
    half_m, half_n = block_m // 2, block_n // 2
    reg_m, reg_n = half_m // warps_m, half_n // warps_n
    warps = warps_m * warps_n
    if warps != 8:
        raise ValueError(f"the pingpong pairs two halves of four warps; "
                         f"{warps_m}x{warps_n} is {warps}")
    if reg_m % 16 or reg_n % 16 or reg_m * warps_m != half_m or reg_n * warps_n != half_n:
        raise ValueError(f"{block_m}x{block_n} over {warps_m}x{warps_n} warps "
                         f"does not give whole 16-row accumulators")
    if k_step % 32:
        raise ValueError(f"k_step={k_step}: the MFMA's operand is 32 deep")

    st_a = hk.ST(bf16, half_m, k_step)
    st_b = hk.ST(bf16, half_n, k_step)
    t_a = hk.RT(bf16, reg_m, k_step, "row")
    t_b = hk.RT(bf16, reg_n, k_step, "row")
    # The MFMA's accumulator holds four consecutive rows of one column in a
    # lane, so storing C from it is one 2-byte store per element.
    # `transposed_c` accumulates Cᵀ = B·Aᵀ instead, whose four values are
    # consecutive columns of a row of C: transposed (a relabel), one 8-byte
    # store per 16x16 fragment, a quarter of the instructions. It is slower:
    # the store costs what its bytes cost, not its instructions, and the
    # swapped MFMA operands cost 1.2% in the loop. Off, and kept as the
    # measurement.
    t_c = (hk.RT(fp32, reg_n, reg_m, "col") if transposed_c
           else hk.RT(fp32, reg_m, reg_n, "col"))
    # A stream-K share is stored transposed (see gemm_sk).
    t_ct = hk.RT(fp32, reg_n, reg_m, "row")

    @hk.func
    def tile(g, As, Bs, A, B, C, a, b, m0, n0, wr, wc, k0, k_count, out, init=None,
             stores=True, even=False):
        """One block of C: the copies, the pingpong over k_count K-tiles from
        k0, and (`stores`) C's store through `out`. The accumulators C are
        the caller's, zeroed or initialized. `even`: k_count is even, so no
        K-tile is a pad and no copy needs a row limit."""
        # The loop takes K-tiles in pairs. An odd count gets one more, past K,
        # copied with a row limit of 0: it reads zeros and its MFMAs add
        # nothing. The limit narrows the descriptor and moves no address --
        # moving the tile's row instead put the 64-bit address arithmetic of
        # every copy back in the loop, 60 VALU an iteration, 7% slower.
        n_tiles = k_count if even else 2 * hk.cdiv(k_count, 2)
        k_tiles = k_count

        # Only the copies of a pair's second tile can be the pad: they take
        # the limit, and the rest no arithmetic at all.
        def limit(t, rows, pad):
            return (1 - hk.min(hk.max(t - k_tiles + 1, 0), 1)) * rows if pad and not even else None

        def copy_a(t, u, h, pad=False):
            g.load(As[2 * u + h], a.at[m0 + h * half_m, k0 + t * k_step], wait=False,
                   rows=limit(t, half_m, pad), nontemporal="a" in nt)

        def copy_b(t, u, h, pad=False):
            g.load(Bs[2 * u + h], b.at[n0 + h * half_n, k0 + t * k_step], wait=False,
                   rows=limit(t, half_n, pad), nontemporal="b" in nt)

        def read_a(u, h):
            As[2 * u + h].subtile(reg_m, k_step, wr, 0).load(t_a, wait=False, into=A)

        def read_b(u, h, s):
            Bs[2 * u + h].subtile(reg_n, k_step, wc, 0).load(t_b, wait=False, into=B[s])

        def mma(i, j):
            if transposed_c:
                C[i][j] += B[j] @ A.T
            else:
                C[i][j] += A @ B[j].T

        def store(i, j):
            if stores:
                tc = hk.transpose(C[i][j]) if transposed_c else C[i][j]
                out(tc, m0 + i * half_m + wr * reg_m, n0 + j * half_n + wc * reg_n)

        def math(i, j):
            if setprio:
                hk.arch.setprio(1)
            mma(i, j)
            if setprio:
                hk.arch.setprio(0)
            hk.arch.barrier()

        # Prologue: K-tile 0 in buffer 0, three quarters of tile 1 in buffer 1.
        copy_b(0, 0, 0)
        copy_a(0, 0, 0)
        copy_b(0, 0, 1)
        copy_a(0, 0, 1)
        if init is not None:
            # The accumulators start as C's tile of an operand (an epilogue's
            # init_from), its loads in flight with the first tiles' copies.
            for i in range(2):
                for j in range(2):
                    init(C[i][j], m0 + i * half_m + wr * reg_m,
                         n0 + j * half_n + wc * reg_n)
        # The two halves are warps 0-3 and 4-7: a SIMD holds waves w and
        # w + 4, so each SIMD has one wave of each. On a 2 x 4 grid that is
        # warp row 1 (HipKittens' test, kept as it is written there).
        with hk.arch.pingpong(wr == 1 if warps_m == 2 else hk.arch.warp_id() // 4 == 1):
            g.wait_for(Bs[0], As[0])
            hk.arch.barrier()
            copy_b(1, 1, 0, pad=True)
            copy_a(1, 1, 0, pad=True)
            copy_b(1, 1, 1, pad=True)
            # Bs[2] too, as at the end of every iteration: the first fourth
            # memory phase reads it before its own wait.
            g.wait_for(*((Bs[1], As[1]) if hipkittens_waits or late_b else (Bs[2],)))
            hk.arch.barrier()

            for t in range(0, n_tiles - 2, 2):
                # K-tile t, buffer 0.
                read_b(0, 0, 0)
                read_a(0, 0)
                copy_a(t + 1, 1, 1)
                hk.lds.wait_for(B[0])
                hk.arch.barrier()
                hk.lds.wait_for(A)
                math(0, 0)
                hk.arch.sched_barrier()

                read_b(0, 1, 1)
                copy_b(t + 2, 0, 0)
                hk.arch.barrier()
                hk.lds.wait_for(B[1])
                math(0, 1)

                read_a(0, 1)
                copy_a(t + 2, 0, 0)
                hk.arch.barrier()
                hk.lds.wait_for(A)
                math(1, 0)
                hk.arch.sched_barrier()

                if not late_b:
                    read_b(1, 0, 0)             # the next K-tile's first B
                copy_b(t + 2, 0, 1)
                g.wait_for(As[3])
                hk.arch.barrier()
                math(1, 1)

                # K-tile t + 1, buffer 1.
                if late_b:
                    read_b(1, 0, 0)
                read_a(1, 0)
                copy_a(t + 2, 0, 1)
                hk.lds.wait_for(B[0])
                hk.arch.barrier()
                hk.lds.wait_for(A)
                math(0, 0)
                hk.arch.sched_barrier()

                read_b(1, 1, 1)
                copy_b(t + 3, 1, 0, pad=True)
                hk.arch.barrier()
                hk.lds.wait_for(B[1])
                math(0, 1)

                read_a(1, 1)
                copy_a(t + 3, 1, 0, pad=True)
                hk.arch.barrier()
                hk.lds.wait_for(A)
                math(1, 0)
                hk.arch.sched_barrier()

                copy_b(t + 3, 1, 1, pad=True)
                # HipKittens waits for all but three copies here, which leaves
                # Bs[2] -- read by the next iteration's fourth memory phase,
                # before its own wait -- covered by nothing but time.
                g.wait_for(As[1] if hipkittens_waits or late_b else Bs[2])
                hk.arch.barrier()
                math(1, 1)

            # The last two K-tiles: nothing left to copy but A's second half.
            t = n_tiles - 2
            read_b(0, 0, 0)
            read_a(0, 0)
            copy_a(t + 1, 1, 1, pad=True)
            hk.arch.barrier()
            hk.lds.wait_for(B[0], A)
            math(0, 0)

            read_b(0, 1, 1)
            hk.arch.barrier()
            hk.lds.wait_for(B[1])
            math(0, 1)

            read_a(0, 1)
            g.wait_for(Bs[3])
            hk.arch.barrier()
            hk.lds.wait_for(A)
            if setprio:
                hk.arch.setprio(1)
            mma(1, 0)
            mma(1, 1)
            if setprio:
                hk.arch.setprio(0)
            hk.arch.barrier()

            read_b(1, 0, 0)
            read_a(1, 0)
            g.wait_for(As[3])
            hk.arch.barrier()
            hk.lds.wait_for(B[0], A)
            math(0, 0)

            read_b(1, 1, 1)
            if early_store:
                store(0, 0)
            hk.arch.barrier()
            hk.lds.wait_for(B[1])
            math(0, 1)

            read_a(1, 1)
            if early_store:
                store(0, 1)
            hk.arch.barrier()
            hk.lds.wait_for(A)
            if setprio:
                hk.arch.setprio(1)
            mma(1, 0)
            mma(1, 1)
            if setprio:
                hk.arch.setprio(0)
            hk.arch.barrier()

        for i in range(2):
            for j in range(2):
                if not (early_store and i == 0):
                    store(i, j)

    def alloc(warps_):
        g = hk.group(warps_)
        # Buffer u, half h of A is As[2u + h]: every index below is a Python
        # int, so each copy and read names its buffer.
        As = hk.lds.alloc(st_a, count=4, name="As", shape=lds_shape)
        Bs = hk.lds.alloc(st_b, count=4, name="Bs", shape=lds_shape)
        A = hk.zeros(t_a)
        B = [hk.zeros(t_b) for _ in range(2)]
        C = [[hk.zeros(t_c) for _ in range(2)] for _ in range(2)]
        return g, As, Bs, A, B, C

    def tile_of(t, m_blocks, n_blocks, WGM):
        """Block t of the grid, WGM block-rows at a time, so a group's B
        panel stays in L2: (row, col)."""
        per_group = WGM * n_blocks
        first_m = (t // per_group) * WGM
        group_m = hk.min(m_blocks - first_m, WGM)
        row = first_m + (t % per_group) % group_m
        col = (t % per_group) // group_m
        return row, col

    @hk.func
    def gemm(a, b, WGM, k0, k_count, out, init=None):
        g, As, Bs, A, B, C = alloc(warps)

        M = a.rows
        m_blocks = hk.cdiv(M, block_m)
        n_blocks = b.rows // block_n
        blocks = m_blocks * n_blocks

        # Workgroups go to the XCDs round robin; renumber them so a chunk of
        # consecutive ids shares an XCD and its L2 (HipKittens'
        # chiplet_transform_chunked). Ids past the last whole round stay.
        wgid = hk.arch.block_idx.x
        per_round = NUM_XCDS * xcd_chunk
        limit = (blocks // per_round) * per_round
        local = wgid // NUM_XCDS
        moved = ((local // xcd_chunk) * per_round + (wgid % NUM_XCDS) * xcd_chunk
                 + local % xcd_chunk)
        stays = hk.min(hk.max(wgid - limit, 0), 1)      # 1 past the last round
        wgid = moved + stays * (wgid - moved)
        # Then WGM block-rows at a time, so a group's B panel stays in L2.
        row, col = tile_of(wgid, m_blocks, n_blocks, WGM)
        m0 = row * block_m
        n0 = col * block_n

        wid = hk.arch.warp_id()
        wr = wid // warps_n
        wc = wid % warps_n
        tile(g, As, Bs, A, B, C, a, b, m0, n0, wr, wc, k0, k_count, out, init)

    @hk.func
    def gemm_sk(a, b, ws, cnt, plan, out):
        """The persistent form: workgroup g runs the segments sk_plan gave it
        -- a block and a range of its K -- in order. A segment that is a
        whole block is stored as it is finished. A block whose K is cut
        between workgroups is stored by each as an fp32 share, and the last
        to finish its share (hk.arch.wg_arrive) adds the shares, in the
        plan's order, after its own last segment: the main loop's registers
        are free then, and the last share of a block is in practice the last
        thing a workgroup does. The plan is read with scalar loads, so the
        loop carries an index, not the arithmetic that made it."""
        g, As, Bs, A, B, C = alloc(warps)
        wid = hk.arch.warp_id()
        wr = wid // warps_n
        wc = wid % warps_n

        def field(r, f):
            return plan[hk.elem_coord(0, 0, 0, r + f)].load_scalar(invariant=True)

        def coords(i, j):
            return i * half_m + wr * reg_m, j * half_n + wc * reg_n

        def store_all(m0, n0):
            for i in range(2):
                for j in range(2):
                    r, cc = coords(i, j)
                    out(hk.transpose(C[i][j]) if transposed_c else C[i][j], m0 + r, n0 + cc)

        wgid = hk.arch.block_idx.x
        start = field(wgid, 0)
        segs = (field(wgid, 1) - start) // SK_REC
        # Which cut blocks it sums, being the last to finish their shares:
        # a workgroup's cut segments are at most its first and its last
        # stream-K one (bits 0 and 1; plan fields 8, and the records named
        # after the offsets).
        fixes = 0
        for s in range(segs):
            r0 = start + s * SK_REC
            m0 = field(r0, 0)
            n0 = field(r0, 1)
            for i in range(2):
                for j in range(2):
                    C[i][j].zero_()
            tile(g, As, Bs, A, B, C, a, b, m0, n0, wr, wc, field(r0, 2),
                 field(r0, 3), out, stores=False, even=True)
            slot = field(r0, 4)
            if slot < 0:
                store_all(m0, n0)
            else:
                # Transposed: an accumulator's four rows a lane are one
                # 16-byte run of a share stored (column, row).
                for i in range(2):
                    for j in range(2):
                        r, cc = coords(i, j)
                        ws[hk.elem_coord(0, slot, cc, r)].store(hk.transpose(C[i][j]))
                n_sh = field(r0, 6)
                arrived = hk.arch.wg_arrive(cnt[hk.elem_coord(0, 0, 0, field(r0, 5))], n_sh,
                                            scope="xcd")
                if arrived == n_sh - 1:
                    fixes = fixes + field(r0, 8)

        for bit in (1, 2):
            if (fixes // bit) % 2 == 1:
                r0 = field(wgid, bit * sk_grid + 1)
                first_slot = field(r0, 7)
                n_sh = field(r0, 6)
                # A quadrant at a time, the shares in the plan's order (the
                # same sum whichever workgroup is last): one share live
                # beside the accumulators.
                for i in range(2):
                    for j in range(2):
                        C[i][j].zero_()
                        r, cc = coords(i, j)
                        for sl in range(first_slot, first_slot + n_sh):
                            C[i][j] += hk.transpose(ws[hk.elem_coord(0, sl, cc, r)].load(t_ct))
                store_all(field(r0, 0), field(r0, 1))

    def body(a: hk.Tensor[bf16], b: hk.Tensor[bf16], c: hk.Tensor[bf16], *,
             WGM: hk.Constexpr = wgm, **ops):
        inited = (epilogue is not None and getattr(epilogue, "init", None)
                  is not None and not transposed_c)

        def out(tile, r, col):
            if epilogue is None:
                c[hk.elem_coord(0, 0, r, col)].store(tile, bounded=True)
            else:
                epilogue.store(c, ops, tile, r, col, inited=inited)

        def init(tile, r, col):
            epilogue.load_init(ops, tile, r, col)

        gemm(a, b, WGM, 0, a.cols // k_step, out, init if inited else None)

    # splits > 1: K cut across workgroups, each share to an fp32 scratch
    # (splits, M, N), and hk.ops.gemm_cdna_dp.reduce_kernel adds them.
    def body_split(a: hk.Tensor[bf16], b: hk.Tensor[bf16], ws: hk.Tensor[fp32], *,
                   WGM: hk.Constexpr = wgm):
        s = hk.arch.block_idx.y
        per = a.cols // splits

        def out(tile, r, col):
            ws[hk.elem_coord(0, s, r, col)].store(tile, bounded=True)

        gemm(a, b, WGM, s * per, per // k_step, out)

    # stream_k: sk_grid workgroups, each running the segments sk_plan gave
    # it; ws holds the fp32 shares of cut blocks (sk_scratch), cnt a counter
    # a block, zero between launches (the last arrival resets it). The
    # epilogue runs once, on a block's whole sum.
    def body_sk(a: hk.Tensor[bf16], b: hk.Tensor[bf16], c: hk.Tensor[bf16],
                ws: hk.Tensor[fp32], cnt: hk.Tensor[i32], plan: hk.Tensor[i32], *,
                SKG: hk.Constexpr = sk_grid, **ops):
        def out(t, r, col):
            if epilogue is None:
                c[hk.elem_coord(0, 0, r, col)].store(t, bounded=True)
            else:
                epilogue.store(c, ops, t, r, col, inited=False)

        gemm_sk(a, b, ws, cnt, plan, out)

    if stream_k:
        if splits != 1:
            raise ValueError("stream_k cuts K itself; splits must be 1")
        if transposed_c:
            raise ValueError("stream_k: a cut block's shares are stored and summed in "
                             "the untransposed accumulators' layout; transposed_c must be False")
        if sk_grid % NUM_XCDS:
            raise ValueError(f"sk_grid={sk_grid}: a whole number of workgroups an XCD")
        if stream_k not in SK_MODES:
            raise ValueError(f"stream_k={stream_k!r}: one of {sorted(SK_MODES)}")
        fn = _epilogue.sign(body_sk, epilogue)
    elif splits == 1:
        fn = _epilogue.sign(body, epilogue)
    elif epilogue is not None:
        raise ValueError("a split kernel stores fp32 shares; its epilogue goes "
                         "on the reduction (gemm_cdna_dp.reduce)")
    else:
        fn = body_split
    fn.__name__ = name

    def _grid(p):
        if stream_k:
            return (p.SKG, 1, 1)
        return (hk.cdiv(p.a.rows, block_m) * (p.b.rows // block_n), splits, 1)

    return hk.kernel(fn, arch=arch, warps=warps, grid=_grid, name=name,
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)


#: HipKittens' tiling.
DEFAULT: Dict[str, Any] = dict(block_m=256, block_n=256, k_step=64, warps_m=2,
                               warps_n=4, wgm=8, xcd_chunk=64, setprio=True,
                               transposed_c=False, early_store=True, late_b=True,
                               splits=1, lds_shape=None, nt="")


def _sched_name(s: Dict[str, Any]) -> str:
    return ("gemm_cdna_{block_m}x{block_n}x{k_step}_w{warps_m}x{warps_n}"
            "_g{wgm}_x{xcd_chunk}").format(**s) + ("_sp" if s.get("setprio") else "") + (
                "_ct" if s.get("transposed_c") else "") + ("_es" if s.get("early_store") else "") + (
                "_lb" if s.get("late_b") else "") + (
                f"_k{s['splits']}" if s.get("splits", 1) > 1 else "") + (
                f"_l{s['lds_shape']}" if s.get("lds_shape") else "") + (
                f"_nt{s['nt']}" if s.get("nt") else "") + (
                f"_sk{s.get('sk_grid', 256)}" if s.get("stream_k") else "")


_KERNELS: Dict[str, Any] = {}


def kernel_for(sched: Dict[str, Any]):
    s = {**DEFAULT, **sched}
    name = _sched_name(s)
    if name not in _KERNELS:
        _KERNELS[name] = gemm_kernel(name, **s)
    return _KERNELS[name]


#: stream_k's partitions of a grid among its sk_grid workgroups: 1 cuts
#: every block's K into equal runs (stream-K proper), 2 deals whole blocks
#: round robin for the grid's whole waves and cuts only the remainder.
SK_MODES = {1: "sk", 2: "dp+sk"}

#: Ints a segment of a stream-K plan takes: m0, n0, k0 (elements), K-tiles,
#: share slot (-1: a whole block, stored as is), block, shares of the block,
#: the block's first slot, and its bit (1: the workgroup's first cut
#: segment, 2: its last).
SK_REC = 9


def sk_plan(m: int, n: int, k: int, block_m: int, block_n: int, k_step: int,
            sk_grid: int, mode: int, wgm: int):
    """A stream-K launch's plan, as a list of ints: sk_grid + 1 offsets (the
    segments of workgroup g are records offsets[g] .. offsets[g + 1]), the
    record of each workgroup's first cut segment and of its last (sk_grid
    each, -1: none), then SK_REC ints a segment; and the number of share
    slots it uses.

    Blocks are numbered as the data-parallel pingpong numbers them (WGM
    block-rows at a time). Mode 2 first deals the grid's whole waves of
    blocks round robin, whole. The blocks left are cut: each XCD takes a
    contiguous eighth of them, whole, and its sk_grid / 8 workgroups split
    those blocks' K-pairs into equal runs -- so every workgroup sharing a
    cut block runs on one XCD and the shares meet in its L2 (wg_arrive
    scope="xcd"), and the blocks an XCD works on are neighbours. A cut
    block's shares take consecutive slots in run order; the last run to
    finish sums them in that order. Workgroup g runs on XCD g % 8."""
    mb, nb = -(-m // block_m), n // block_n
    blocks = mb * nb
    per = k // (2 * k_step)                    # units a block: K-tile pairs
    per_x = sk_grid // NUM_XCDS

    def origin(t):
        per_group = wgm * nb
        first_m = (t // per_group) * wgm
        group_m = min(mb - first_m, wgm)
        return ((first_m + (t % per_group) % group_m) * block_m,
                ((t % per_group) // group_m) * block_n)

    segs = [[] for _ in range(sk_grid)]        # by workgroup: (block, unit lo, unit hi)
    whole = (blocks // sk_grid) * sk_grid if mode == 2 else 0
    for t in range(whole):
        # Consecutive blocks to one XCD's workgroups, a wave at a time.
        i = t % sk_grid
        segs[(i % per_x) * NUM_XCDS + i // per_x].append((t, 0, per))
    rest = blocks - whole
    nsk = [0] * sk_grid                        # by workgroup: its stream-K segments
    for x in range(NUM_XCDS):
        b0, b1 = whole + x * rest // NUM_XCDS, whole + (x + 1) * rest // NUM_XCDS
        units = (b1 - b0) * per
        for j in range(per_x):
            lo, hi = j * units // per_x, (j + 1) * units // per_x
            u = lo
            while u < hi:
                t = u // per
                end = min(hi, (t + 1) * per)
                segs[j * NUM_XCDS + x].append((b0 + t, u - t * per, end - t * per))
                nsk[j * NUM_XCDS + x] += 1
                u = end
    # A run's cut segments in reverse: the head of its last block first, so
    # every run of an XCD starts its stream-K work at K = 0 and they read
    # the same K-tiles of A and B together (in run order, each started at
    # its own K and an XCD's L2 kept almost nothing for its neighbours: a
    # 1024x5120x7168 GEMM in 128x256 blocks took 114 us, head first 89.5).
    segs = [ss[:len(ss) - nsk[g]] + ss[len(ss) - nsk[g]:][::-1] for g, ss in enumerate(segs)]
    # Shares: a cut block's workgroups in run order, consecutive slots.
    sharers: Dict[int, list] = {}
    for x in range(NUM_XCDS):
        for j in range(per_x):
            g = j * NUM_XCDS + x
            for t, ulo, uhi in segs[g]:
                if uhi - ulo < per:
                    sharers.setdefault(t, []).append(g)
    first_slot, slots = {}, 0
    for t in sorted(sharers):
        first_slot[t] = slots
        slots += len(sharers[t])
    recs, cuts = [], []
    for g, ss in enumerate(segs):
        cut = [i for i, (t, _, _) in enumerate(ss) if t in sharers]
        if len(cut) > 2 or (len(cut) == 2 and any(t in sharers for t, _, _ in ss[cut[0] + 1:cut[1]])):
            raise AssertionError(f"workgroup {g}: cut segments {cut}")
        bit = {i: 1 + (i != cut[0]) for i in cut}
        cuts.append(cut)
        out = []
        for i, (t, ulo, uhi) in enumerate(ss):
            m0, n0 = origin(t)
            if t in sharers:
                sh = sharers[t]
                out += [m0, n0, ulo * 2 * k_step, (uhi - ulo) * 2,
                        first_slot[t] + sh.index(g), t, len(sh), first_slot[t], bit[i]]
            else:
                out += [m0, n0, 0, 2 * per, -1, t, 1, -1, 0]
        recs.append(out)
    offsets, at = [], 3 * sk_grid + 1
    for r in recs:
        offsets.append(at)
        at += len(r)
    offsets.append(at)
    fix = [[-1] * sk_grid, [-1] * sk_grid]
    for g, cut in enumerate(cuts):
        for i in cut:
            fix[1 + (i != cut[0]) - 1][g] = offsets[g] + i * SK_REC
    return offsets + fix[0] + fix[1] + [x for r in recs for x in r], slots


_SK: Dict[Any, Any] = {}


def sk_args(sched: Dict[str, Any], device, m: int, n: int, k: int):
    """The extra arguments (ws, cnt, plan) of a schedule's launch at
    M x N x K: () unless it is stream-K. Made once per shape and shared by
    every launch (launches on one stream run in order; the last arrival at
    a block resets its counter, so cnt is zero again when the next launch
    starts)."""
    mode = sched.get("stream_k", 0)
    if not mode:
        return ()
    import torch  # noqa: PLC0415

    s = {**DEFAULT, **sched}
    bm, bn, ks, grid = s["block_m"], s["block_n"], s["k_step"], s.get("sk_grid", 256)
    key = (str(device), m, n, k, bm, bn, ks, grid, mode, s["wgm"])
    got = _SK.get(key)
    if got is None:
        flat, slots = sk_plan(m, n, k, bm, bn, ks, grid, mode, s["wgm"])
        blocks = -(-m // bm) * (n // bn)
        got = _SK[key] = (
            torch.empty((max(slots, 1), bn, bm), device=device, dtype=torch.float32),
            torch.zeros((max(blocks, 1),), device=device, dtype=torch.int32),
            torch.tensor(flat, device=device, dtype=torch.int32))
    return got


#: What every schedule takes: K whole 64-wide tiles (an odd count is padded
#: with a tile of zeros), N whole blocks of the narrowest block_n, any M.
K_STEP, BLOCK_N_MIN = 64, 128


def schedule_for(m: int, n: int, k: int) -> Dict[str, Any]:
    """The schedule for a shape: 256-wide blocks of N where they tile it, and
    the block of M no taller than needed (a partial block computes rows that
    are never stored)."""
    s = dict(DEFAULT)
    if n % 256:
        s["block_n"] = 128
    # 128 is the least: half a block is then 64 rows by 64, 8 KB, one 16-byte
    # copy for each of the 512 threads.
    if m <= 128:
        s["block_m"] = 128
    return s


#: hk's own gfx950 GEMMs a shape is routed among: this module's pingpong for
#: large M, hk.ops.gemm_cdna_dp's four-warp kernel for the M it has too few
#: blocks at, and hk.ops.decode_gemm (W straight to registers) for decode
#: batches. tools/hk-bench/gemm_cdna_routes.py measures them.
CANDIDATES: Dict[str, Any] = {
    "pp": ("pp", {}),
    "ppk2": ("pp", {"splits": 2}),
    "ppk4": ("pp", {"splits": 4}),
    "dp128": ("dp", {"block_m": 128, "block_n": 128, "stages": 4}),
    "dp128k2": ("dp", {"block_m": 128, "block_n": 128, "stages": 4, "splits": 2}),
    "dp128k4": ("dp", {"block_m": 128, "block_n": 128, "stages": 4, "splits": 4}),
    "dp64": ("dp", {"block_m": 64, "block_n": 128, "stages": 3, "xcd_chunk": 0}),
    "dp64k4": ("dp", {"block_m": 64, "block_n": 128, "stages": 3, "xcd_chunk": 0, "splits": 4}),
    "dp64k8": ("dp", {"block_m": 64, "block_n": 128, "stages": 3, "xcd_chunk": 0, "splits": 8}),
    "dp64x64": ("dp", {"block_m": 64, "block_n": 64, "stages": 4, "xcd_chunk": 0}),
    "dp64x64k2": ("dp", {"block_m": 64, "block_n": 64, "stages": 4, "xcd_chunk": 0, "splits": 2}),
    "dp64x64k4": ("dp", {"block_m": 64, "block_n": 64, "stages": 4, "xcd_chunk": 0, "splits": 4}),
    "dp64x64k8": ("dp", {"block_m": 64, "block_n": 64, "stages": 4, "xcd_chunk": 0, "splits": 8}),
    "pp128": ("pp", {"block_m": 128}),
    "decode": ("decode", {}),
    # Blocks that put a whole number of workgroups on each of the 256 CUs
    # where a power of two does not: 96 x 256 = 24576, 224 x 128 = 28672.
    "dp64x96": ("dp", {"block_m": 64, "block_n": 96, "stages": 4, "xcd_chunk": 0}),
    "dp128x96": ("dp", {"block_m": 128, "block_n": 96, "stages": 4, "xcd_chunk": 0}),
    "dp128x224": ("dp", {"block_m": 128, "block_n": 224, "stages": 3, "xcd_chunk": 0}),
    "dp128x64": ("dp", {"block_m": 128, "block_n": 64, "stages": 4, "xcd_chunk": 0}),
    "dp128x0": ("dp", {"block_m": 128, "block_n": 128, "stages": 4, "xcd_chunk": 0}),
    "dp128x0k4": ("dp", {"block_m": 128, "block_n": 128, "stages": 4, "xcd_chunk": 0,
                         "splits": 4}),
    "dp128k8": ("dp", {"block_m": 128, "block_n": 128, "stages": 4, "xcd_chunk": 0,
                       "splits": 8}),
    "dp64k128": ("dp", {"block_m": 64, "block_n": 128, "k_step": 128, "stages": 3,
                        "xcd_chunk": 0}),
    # Weights streamed: whole-line copies (st_16x64) past the caches (sc0 nt),
    # 128-256 deep a K-tile -- what the paged decode's copies took.
    "dp32x96nt": ("dp", {"block_m": 32, "block_n": 96, "k_step": 256, "stages": 2,
                         "xcd_chunk": 0, "lds_shape": "16x64", "nt": "b"}),
    "dp32x64nt": ("dp", {"block_m": 32, "block_n": 64, "k_step": 256, "stages": 2,
                         "xcd_chunk": 0, "lds_shape": "16x64", "nt": "b"}),
    "dp32nt": ("dp", {"block_m": 32, "block_n": 128, "k_step": 128, "stages": 3,
                      "xcd_chunk": 0, "lds_shape": "16x64", "nt": "b"}),
    "dp64nt": ("dp", {"block_m": 64, "block_n": 128, "k_step": 128, "stages": 3,
                      "xcd_chunk": 0, "lds_shape": "16x64", "nt": "b"}),
    "dp32x96ntk4": ("dp", {"block_m": 32, "block_n": 96, "k_step": 256, "stages": 2,
                           "xcd_chunk": 0, "lds_shape": "16x64", "nt": "b", "splits": 4}),
    # Two warps of 32 x 48: 256 blocks at N = 12288, one a CU, 5.7 TB/s at
    # M = 32 against hipBLASLt's 4.8.
    "dp32x48nt": ("dp", {"block_m": 32, "block_n": 48, "k_step": 128, "stages": 4,
                         "warps_m": 2, "warps_n": 1, "xcd_chunk": 0, "lds_shape": "16x64",
                         "nt": "b"}),
    "ppnt": ("pp", {"lds_shape": "16x64", "nt": "b"}),
    # The pingpong on the 256x256 blocks that fill whole waves of the CUs,
    # the rest of N another candidate's: 2048 x 12288 is 384 blocks, a wave
    # and a half, 0.915 to 0.983x hipBLASLt with the last 4096 columns as
    # 128-row blocks.
    "ppw": ("ppw", {"rem": "pp128"}),
    "ppwd": ("ppw", {"rem": "dp128x0"}),
}

_ROUTES: Dict[str, Dict[int, str]] = {}

#: Candidates a route table defines that CANDIDATES does not: name ->
#: hk.schedules GemmSchedule, taken exactly as its record says (a tuner's
#: schedules, `[family, {knobs}]` in the table's "candidates").
_EXTRA: Dict[str, Any] = {}


def _routes() -> Dict[str, Dict[int, str]]:
    """The measured table, NxK -> {M: candidate}, read once."""
    if not _ROUTES:
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        p = Path(__file__).resolve().parents[1] / "tuned" / "routes_gemm_gfx950.json"
        if p.exists():
            doc = json.loads(p.read_text())
            extra = {nm: c for nm, c in doc.get("candidates", {}).items()
                     if nm not in CANDIDATES}
            if extra:
                from .. import schedules  # noqa: PLC0415

                for nm, c in extra.items():
                    _EXTRA[nm] = schedules.from_record(c)
            for nk, by_m in doc.get("routes", {}).items():
                _ROUTES[nk] = {int(m): c for m, c in by_m.items()
                               if c in CANDIDATES or c in _EXTRA}
        _ROUTES.setdefault("", {})
    return _ROUTES


def extra_candidates() -> Dict[str, Any]:
    """The route table's own candidates (see _EXTRA)."""
    _routes()
    return _EXTRA


def fits(name: str, m: int, n: int, k: int) -> bool:
    """Whether candidate `name` takes the shape at all -- not whether it wins."""
    if name not in CANDIDATES:
        s = extra_candidates()[name]
        return n % BLOCK_N_MIN == 0 and k % K_STEP == 0 and not s.check(m, n, k, "gfx950")
    kind, sched = CANDIDATES[name]
    splits = sched.get("splits", 1)
    if m < 1 or k % (sched.get("k_step", K_STEP) * splits) or n % BLOCK_N_MIN:
        return False
    if kind == "dp" and n % sched.get("block_n", 128):
        return False
    # A split's fp32 shares are a scratch tensor per shape, kept: past a GiB
    # of it the shape has blocks enough not to need splitting.
    if splits > 1 and m * n * splits * 4 > 2**30:
        return False
    if kind == "decode":
        from . import decode_gemm  # noqa: PLC0415

        return decode_gemm.fits(m, n, k)
    if kind == "ppw":
        cut = wave_cut(m, n)
        return 0 < cut < n and fits(sched["rem"], m, n - cut, k)
    return True


#: CUs a wave of one-a-CU workgroups fills: an MI355X's.
CUS = 256


def wave_cut(m: int, n: int) -> int:
    """The columns of N whose 256x256 blocks are whole waves of CUS
    workgroups at M's block-rows: 0 where a wave takes more than N has."""
    mb = -(-m // 256)
    if mb > CUS:
        return 0
    per = CUS // mb
    return (n // 256 // per) * per * 256


def user_route(m: int, n: int, k: int) -> Optional[str]:
    """This machine's hk.tune record for the shape's M bucket
    (hk.tune.records), as a candidate name of _EXTRA, if there is one and it
    takes the shape; else None."""
    from ..autotune import generation  # noqa: PLC0415

    key = (m, n, k)
    gen = generation()
    hit = _USER.get(key)
    if hit is not None and hit[0] == gen:
        return hit[1]
    from ..tune import records  # noqa: PLC0415

    name = None
    u = records.user_route("gfx950", m, n, k)
    if u is not None:
        try:
            from .. import schedules  # noqa: PLC0415

            s = schedules.from_record(u[1], strict=False)
            if not s.check(m, n, k, "gfx950"):
                name = u[0]
                _EXTRA[name] = s
        except (ValueError, TypeError, KeyError):
            name = None                 # an unreadable record costs speed, not a launch
    if len(_USER) > 4096:
        _USER.clear()
    _USER[key] = (gen, name)
    return name


#: user_route's answers, by shape, at the records generation they were made
#: at: route() is on hk.ops.matmul's launch path.
_USER: Dict[tuple, tuple] = {}


def route(m: int, n: int, k: int, user: bool = True) -> str:
    """The candidate for a shape: this machine's hk.tune record for its M
    bucket (`user=False` skips it) if there is one and it takes the shape;
    else the measured table's, at the nearest M it measured for this N x K
    (on a log scale), if that candidate takes the shape; else the rule the
    table's measurements mostly follow."""
    if user:
        _routes()
        name = user_route(m, n, k)
        if name is not None:
            return name
    by_m = _routes().get(f"{n}x{k}")
    if by_m:
        near = min(by_m, key=lambda x: (abs((x / m) if x > m else (m / x)), x))
        if fits(by_m[near], m, n, k):
            return by_m[near]
    if m <= 32 and fits("decode", m, n, k):
        return "decode"
    if m <= 512:
        return "dp64"
    if m <= 2048:
        return "dp128"
    return "pp"


#: The kernel for the pingpong's last 128 columns, by M. As one 128-wide
#: pingpong block a 256 rows they are a handful of workgroups each walking
#: all of K, 41-50 us at K = 4096 on an MI355X; split along K in small
#: blocks, 8-20 us (M = 256-8192).
TAIL: tuple = ((1024, "dp64x64k8"), (2048, "dp64x64k4"), (4096, "dp64k4"),
               (None, "dp64x64"))


def tail_for(m: int, k: int) -> str:
    """The candidate for an M x 128 x K tail."""
    for bound, name in TAIL:
        if (bound is None or m <= bound) and fits(name, m, 128, k):
            return name
    return "pp128"


def run(name: str, a, b, out):
    """Candidate `name` on (a, b) into out; the shape checks are matmul's."""
    if name not in CANDIDATES:
        return extra_candidates()[name].launch(a, b, out)
    kind, sched = CANDIDATES[name]
    if kind == "decode":
        from . import decode_gemm  # noqa: PLC0415

        return decode_gemm.matmul(a, b, out=out)
    if kind == "dp":
        from . import gemm_cdna_dp  # noqa: PLC0415

        return gemm_cdna_dp.launch(sched, a, b, out)
    m, k = a.shape
    n = b.shape[0]
    if kind == "ppw":
        cut = wave_cut(m, n)
        run("pp", a, b[:cut], out[:, :cut])
        run(sched["rem"], a, b[cut:], out[:, cut:])
        return out
    if sched.get("splits", 1) > 1:
        from . import gemm_cdna_dp  # noqa: PLC0415

        s = {**schedule_for(m, n, k), **sched}
        ws = gemm_cdna_dp.scratch_for(a.device, s["splits"], m, n)
        kernel_for(s)(a, b, ws)
        gemm_cdna_dp.reduce(ws, out)
        return out
    wide = n - n % 256
    if n % 256 and wide:
        # A 128-wide block runs at two thirds of a 256-wide one, so only the
        # last 128 columns are apart: two launches into column slices of out,
        # which a strided global takes as they are (lm_head's 151936).
        kernel_for({**schedule_for(m, wide, k), **sched})(a, b[:wide], out[:, :wide])
        run(tail_for(m, k), a, b[wide:], out[:, wide:])
        return out
    kernel_for({**schedule_for(m, n, k), **sched})(a, b, out)
    return out


def matmul(a, b, out=None):
    """A @ B.T for bf16 on gfx950, fp32 accumulate; `b` is (N, K), as in
    hk.ops.gemm.matmul, whose gfx950 form this is. The kernel is one of hk's
    own, by route()."""
    import torch  # noqa: PLC0415 -- hk traces and compiles without torch

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"matmul: want 2D, got {a.dim()}D and {b.dim()}D")
    m, k = a.shape
    n, kb = b.shape
    if kb != k:
        raise ValueError(f"matmul: A is (M={m}, K={k}) and B is (N={n}, K={kb}); "
                         f"B is passed pre-transposed, so its second axis is K")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"matmul: bf16 only for now, got {a.dtype} and {b.dtype}")
    if k % K_STEP or n % BLOCK_N_MIN or k == 0:
        raise ValueError(
            f"matmul: K={k} must be a multiple of {K_STEP} and N={n} a multiple "
            f"of {BLOCK_N_MIN}. Only M may be ragged: rows of the last block past "
            f"M are loaded as zeros and not stored.")
    if m < 1:
        raise ValueError(f"matmul: M={m}; there is nothing to multiply")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    elif out.shape != (m, n) or out.dtype != torch.bfloat16:
        raise ValueError(f"matmul: out is {tuple(out.shape)}/{out.dtype}, "
                         f"expected ({m}, {n})/bfloat16")
    run(route(m, n, k), a, b, out)
    return out


__all__ = ["gemm_kernel", "kernel_for", "schedule_for", "matmul", "route", "run",
           "fits", "CANDIDATES", "DEFAULT", "K_STEP", "BLOCK_N_MIN"]
