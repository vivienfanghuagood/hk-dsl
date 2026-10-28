"""MXFP8 GEMM on gfx950: C = (sa . A)(sb . B)^T in bf16.

A (M, K) and B (N, K) are fp8 e4m3, K-contiguous (F.linear's layout); sa (M,
K/32) and sb (N, K/32) their E8M0 scales, one byte a row and 32 columns (the
OCP MX layout -- a 32x32-block weight scale, as DeepSeek-V4.1 ships, is its
rows repeated: `expand_block_scales`). On v_mfma_scale_f32_16x16x128_f8f6f4
through `hk.mma_ABt_mx`.

The first form: 128 x 128 blocks of C, four warps of 64 x 64, K in 128-wide
steps through one LDS buffer each for A and B, scales read from global per
step. M and N multiples of 128, K of 128.
"""

from __future__ import annotations

import math

from .. import dsl as hk
from ..mx import aiter_scales, expand_block_scales, pack_scales  # noqa: F401 -- the layouts are hk.mx's
from ..types import bf16, fp32, fp8e4m3, i8, i32

BM = BN = 128
BK = 128
WARPS = 4


def mxfp8_kernel(name: str = "mxfp8_gemm_128x128"):
    st_a = hk.ST(fp8e4m3, BM, BK)
    st_b = hk.ST(fp8e4m3, BN, BK)
    t_a = hk.RT(fp8e4m3, 64, BK, "row")
    t_b = hk.RT(fp8e4m3, 64, BK, "row")
    t_c = hk.RT(fp32, 64, 64, "col")

    def body(a: hk.Tensor[fp8e4m3], b: hk.Tensor[fp8e4m3], sa: hk.Tensor[i8],
             sb: hk.Tensor[i8], c: hk.Tensor[bf16]):
        g = hk.group(WARPS)
        As = hk.lds.alloc(st_a, name="As")
        Bs = hk.lds.alloc(st_b, name="Bs")
        row0 = hk.arch.block_idx.y * BM
        col0 = hk.arch.block_idx.x * BN
        wid = hk.arch.warp_id()
        wr, wc = wid // 2, wid % 2
        acc = hk.zeros(t_c)
        for k0 in range(0, a.cols, BK):
            g.load(As, a.at[row0, k0])
            g.load(Bs, b.at[col0, k0])
            s_a = hk.load_mx_scales(sa, 64, BK // 32, row0 + wr * 64, k0 // 32)
            s_b = hk.load_mx_scales(sb, 64, BK // 32, col0 + wc * 64, k0 // 32)
            g.wait_for(As, Bs)
            hk.arch.barrier()
            ta = As.subtile(64, BK, wr, 0).load(t_a)
            tb = Bs.subtile(64, BK, wc, 0).load(t_b)
            hk.mma("mma_ABt", ta.scaled(s_a), tb.scaled(s_b), acc, into=acc)
            hk.arch.barrier()
        c.at[row0 + wr * 64, col0 + wc * 64] = acc

    body.__name__ = name
    return hk.kernel(body, arch="gfx950", warps=WARPS, name=name,
                     grid=lambda p: (p.c.cols // BN, p.c.rows // BM, 1))


_KERNEL = None


def gemm(a, b, sa, sb, out=None):
    """bf16 (M, N) = (sa . a)(sb . b)^T; see the module docstring."""
    import torch  # noqa: PLC0415

    global _KERNEL
    m, k = a.shape
    n = b.shape[0]
    if m % BM or n % BN or k % BK:
        raise ValueError(f"mxfp8.gemm: M={m} and N={n} multiples of {BM}, K={k} of {BK}")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    if _KERNEL is None:
        _KERNEL = mxfp8_kernel()
    _KERNEL(a, b, sa.view(torch.int8), sb.view(torch.int8), out)
    return out


_PP = {}


def gemm_pp(a, b, sa, sb, out=None, *, block_n: int = 256, **sched):
    """The same product on gfx950's pingpong (hk.ops.gemm_cdna, mx=True):
    256 x block_n blocks of eight warps, K-tiles of 128 double-buffered
    through LDS. N a multiple of block_n, K of 128; any M."""
    import torch  # noqa: PLC0415

    from . import gemm_cdna  # noqa: PLC0415

    m, k = a.shape
    n = b.shape[0]
    if n % block_n or k % 128:
        raise ValueError(f"mxfp8.gemm_pp: N={n} a multiple of {block_n}, K={k} of 128")
    key = (block_n, tuple(sorted(sched.items())))
    kern = _PP.get(key)
    if kern is None:
        name = "mxfp8_pp_" + "_".join([str(block_n)] + [f"{k_}{v}" for k_, v in key[1]])
        kern = _PP[key] = gemm_cdna.gemm_kernel(name, mx=True, k_step=128,
                                                block_n=block_n, arch="gfx950", **sched)
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    if sched.get("mx_scales") == "direct":
        kern(a, b, sa, sb.view(torch.int8), out)
    elif sched.get("mx_scales") == "direct32" and sa.dtype == torch.int32:
        raise ValueError("gemm_pp(mx_scales='direct32'): sa in aiter_scales' layout (uint8)")
    else:
        kern(a, b, sa.view(torch.int8), sb.view(torch.int8), out)
    return out


def gemm_8w(a, b, sa_packed, sb_aiter, out=None, *, block_n: int = 256, **sched):
    """gemm_pp with scales that need no VALU (gemm_cdna mx_scales="direct"): eight warps,
    HipKittens' pingpong; sa_packed = pack_scales(sa), sb_aiter = aiter_scales(sb). With
    block_m=128 sa_packed is aiter_scales(sa) instead (mx_scales="direct32")."""
    mode = "direct32" if sched.get("block_m", 256) == 128 else "direct"
    return gemm_pp(a, b, sa_packed, sb_aiter, out, block_n=block_n, mx_scales=mode, **sched)








def mx4w_kernel(name: str = "mxfp8_4w", *, block_m: int = 256, block_n: int = 256,
                stages: int = 2, wgm: int = 8, xcd_chunk: int = 32, setprio: bool = False,
                split_m: int = 2, split_n: int = 2, scale_ahead: int = 2,
                ilv: int = 0, rolled: bool = False, dbg: str = "", defines: str = "",
                exact: bool = False):
    """MXFP8 GEMM, four warps (2 x 2) of block_m/2 x block_n/2 each, so one
    wave a SIMD and the whole register file a lane (VGPRs and AGPRs: the
    accumulators of a 128 x 128 warp tile are 256 of them). K-tiles of 128 in
    a ring of `stages` LDS buffers, `stages - 1` ahead; a warp's tile is four
    quadrants, its operands read a half at a time under the previous
    quadrant's MFMAs. Scales packed (`pack_scales`), one dword a lane per 64
    rows and K-tile, loaded two K-tiles ahead. Takes (a, b, sa, sb, c): a (M,
    K) and b (N, K) fp8 e4m3, sa / sb their packed scales, c (M, N) bf16; N a
    multiple of block_n, K of 128. `split_m` x `split_n`: the parts of a warp's
    tile (1 or 2 each way), each a multiple of 64 rows of A and of B."""
    BKs = 128
    qm, qn = block_m // 2 // split_m, block_n // 2 // split_n   # a part of a warp's tile
    if split_m not in (1, 2) or split_n not in (1, 2) or qm % 64 or qn % 64:
        raise ValueError(f"{block_m}x{block_n} split {split_m}x{split_n}: parts of "
                         f"64-row multiples (packed scales)")
    # Parts in an order where each shares an operand with the one before.
    order = [p for p in ((0, 0), (0, 1), (1, 1), (1, 0)) if p[0] < split_m and p[1] < split_n]
    # A and B are staged as two halves of rows each (A's half h of buffer u
    # is As[2u + h]), so a K-tile's copy goes out in four pieces, one after
    # each part's reads, under that part's MFMAs (`ilv` > 0: the scheduler is
    # asked for `ilv` MFMAs between two of a piece's loads).
    # An operand split one way only (split 1) is one piece: both its warps'
    # rows would straddle the halves.
    ha, hb = split_m, split_n                              # pieces of A and of B
    st_a = hk.ST(fp8e4m3, block_m // ha, BKs)
    st_b = hk.ST(fp8e4m3, block_n // hb, BKs)
    loads_a = block_m // ha * BKs // (WARPS * 64 * 16)     # 16-byte copies a lane, a piece
    loads_b = block_n // hb * BKs // (WARPS * 64 * 16)
    t_a = hk.RT(fp8e4m3, qm, BKs, "row")
    t_b = hk.RT(fp8e4m3, qn, BKs, "row")
    t_c = hk.RT(fp32, qm, qn, "col")
    # scale_ahead + 1 scale sets, a K-tile's loaded scale_ahead K-tiles
    # ahead, so the body is a whole number of both the LDS ring and the
    # scale rotation.
    if rolled:
        scale_ahead = 1             # two sets: the current K-tile's and the next one's
    if scale_ahead not in (1, 2):
        raise ValueError(f"scale_ahead={scale_ahead}: 1 or 2")
    nsets = scale_ahead + 1
    body_len = stages * nsets // math.gcd(stages, nsets)

    def body(a: hk.Tensor[fp8e4m3], b: hk.Tensor[fp8e4m3], sa: hk.Tensor[i32],
             sb: hk.Tensor[i32], c: hk.Tensor[bf16], *, WGM: hk.Constexpr = wgm):
        g = hk.group(WARPS)
        # One ring of `stages` buffers a piece: a buffer index is then the
        # ring position alone, which `rolled` makes a runtime (rot + r) % stages.
        As = [hk.lds.alloc(st_a, count=stages, name=f"As{h}") for h in range(ha)]
        Bs = [hk.lds.alloc(st_b, count=stages, name=f"Bs{h}") for h in range(hb)]
        A = [hk.zeros(t_a) for _ in range(split_m)]
        B = [hk.zeros(t_b) for _ in range(split_n)]
        C = [[hk.zeros(t_c) for _ in range(split_n)] for _ in range(split_m)]

        M = a.rows
        m_blocks = hk.cdiv(M, block_m)
        n_blocks = b.rows // block_n
        # block_idx.x is already renumbered, chunks of xcd_chunk consecutive ids
        # an XCD (hk.Placement.chunked, installed by hk.dsl.placement).
        wgid = hk.arch.block_idx.x
        per_group = WGM * n_blocks
        first_m = (wgid // per_group) * WGM
        group_m = hk.min(m_blocks - first_m, WGM)
        m0 = (first_m + (wgid % per_group) % group_m) * block_m
        n0 = ((wgid % per_group) // group_m) * block_n

        wid = hk.arch.warp_id()
        wr, wc = wid // 2, wid % 2
        k_tiles = a.cols // BKs
        n_tiles = body_len * hk.cdiv(k_tiles, body_len)

        def rows_left(t, rows):
            return (1 - hk.min(hk.max(t - k_tiles + 1, 0), 1)) * rows

        def piece(t, u, kind, h, pad=True):
            if kind == "a":
                g.load(As[h][u], a.at[m0 + h * (block_m // ha), t * BKs], wait=False,
                       rows=rows_left(t, block_m // ha) if pad else None)
            else:
                g.load(Bs[h][u], b.at[n0 + h * (block_n // hb), t * BKs], wait=False,
                       rows=rows_left(t, block_n // hb) if pad else None)

        PIECES = [p for p in (("a", 0), ("b", 0), ("a", 1), ("b", 1))
                  if p[1] < (ha if p[0] == "a" else hb)]

        def copy(t, u, pad=True):
            for kind, h in PIECES:
                piece(t, u, kind, h, pad)

        # Rows of A in part i of warp wr (B alike): i * (block_m / split_m) +
        # wr * qm -- piece i's rows wr * qm.. .
        def a_row(i):
            return i * (block_m // split_m) + wr * qm

        def b_row(j):
            return j * (block_n // split_n) + wc * qn

        S = [None] * nsets

        def load_scales(q, t):
            old = S[q]
            S[q] = ([hk.load_mx_scales(sa, qm, 4, m0 + a_row(i), t * 4,
                                       into=old[0][i] if old else None) for i in range(split_m)],
                    [hk.load_mx_scales(sb, qn, 4, n0 + b_row(j), t * 4,
                                       into=old[1][j] if old else None) for j in range(split_n)])

        def read_a(u, i):             # part i is piece i (or the one piece), row wr * qm
            As[i][u].subtile(qm, BKs, wr, 0).load(t_a, wait=False, into=A[i])

        def read_b(u, j):
            Bs[j][u].subtile(qn, BKs, wc, 0).load(t_b, wait=False, into=B[j])

        def mma(q, i, j):
            if setprio:
                hk.arch.setprio(1)
            hk.mma("mma_ABt", A[i].scaled(S[q][0][i]), B[j].scaled(S[q][1][j]), C[i][j],
                   into=C[i][j])
            if setprio:
                hk.arch.setprio(0)

        # Scales go out before a K-tile's copy (hk-lds-pipeline never counts
        # loads into registers as the ones a vmcnt leaves in flight: they
        # retire out of order with copies into LDS on gfx950). The last set
        # is made here too, for the loop to load into; its load is repeated
        # by the first K-tile.
        if rolled:
            load_scales(0, 0)
            load_scales(1, 1)          # made here for the loop to load into
            for t in range(stages - 1):
                copy(t, t, pad=False)
        else:
            for q in range(nsets):
                load_scales(q, q)
            for t in range(stages - 1):
                copy(t, t, pad=t > 0)

        def k_tile(t, j, copies, u=None, un=None, q=None, qn_=None, scales=True):
            if u is None:
                u, q = j % stages, j % nsets
                un, qn_ = (u + stages - 1) % stages, (j + scale_ahead) % nsets
            g.wait_for(*[As[h][u] for h in range(ha)], *[Bs[h][u] for h in range(hb)], exact=exact)
            hk.arch.barrier()
            if scales:
                load_scales(qn_, t + scale_ahead)
            nxt = t + stages - 1
            have = set()
            for p, (i, jj) in enumerate(order):
                new = [x for x in (("a", i), ("b", jj)) if x not in have]
                for kind, idx in new:
                    (read_a if kind == "a" else read_b)(u, idx)
                    have.add((kind, idx))
                mine = [PIECES[k] for k in range(len(PIECES)) if k % len(order) == p] if copies else []
                for kind, h in mine:
                    piece(nxt, un, kind, h)
                if new:
                    hk.lds.wait_for(*[A[idx] if kind == "a" else B[idx] for kind, idx in new])
                mma(q, i, jj)
                if ilv and mine:
                    n_vm = sum(loads_a if kind == "a" else loads_b for kind, _ in mine)
                    for _ in range(n_vm):
                        hk.arch.sched_group_barrier("mma", ilv, p + 1)
                        hk.arch.sched_group_barrier("vmem", 1, p + 1)
                if "sbpart" in dbg:
                    hk.arch.sched_barrier()
            # Keep each K-tile's MFMAs in it: sunk past the next K-tile's reads
            # they hold two sets of operands live (spills at 128 x 256).
            hk.arch.sched_barrier()

        if not rolled:
            for t0 in range(0, n_tiles - body_len, body_len):
                for j in range(body_len):
                    k_tile(t0 + j, j, True)
            t0 = n_tiles - body_len
            for j in range(body_len):
                k_tile(t0 + j, j, j + stages - 1 < body_len)
        else:
            # One K-tile a loop iteration: the ring's roles rotate at run time
            # (rot), the scales of the K-tile after this one go into set 1 and
            # are copied into set 0 at the end. A two-K-tile body made the
            # compiler permute the 256 AGPR accumulators at the loop edge.
            rot = 0
            for t in range(0, k_tiles - (stages - 1)):
                k_tile(t, 0, True, u=rot % stages, un=(rot + stages - 1) % stages, q=0, qn_=1)
                for i in range(split_m):
                    S[1][0][i].copy(into=S[0][0][i])
                for jj in range(split_n):
                    S[1][1][jj].copy(into=S[0][1][jj])
                rot = (rot + 1) % stages
            for d in range(stages - 1):
                last = d == stages - 2
                k_tile(k_tiles - (stages - 1) + d, 0, False, u=(rot + d) % stages, un=None,
                       q=0, qn_=1, scales=not last)
                if not last:
                    for i in range(split_m):
                        S[1][0][i].copy(into=S[0][0][i])
                    for jj in range(split_n):
                        S[1][1][jj].copy(into=S[0][1][jj])

        for i in range(split_m):
            for j in range(split_n):
                c[hk.elem_coord(0, 0, m0 + a_row(i), n0 + b_row(j))].store(C[i][j], bounded=True)

    body.__name__ = name
    return hk.kernel(body, arch="gfx950", warps=WARPS, name=name, max_vgprs=512,
                     defines={d: 1 for d in defines.split(",") if d},
                     grid=lambda p: (hk.cdiv(p.a.rows, block_m) * (p.b.rows // block_n), 1, 1),
                     placement=hk.Placement.chunked(xcd_chunk, "stay"))


def mx4p_kernel(name: str = "mxfp8_4p", *, block_m: int = 256, block_n: int = 256,
                stages: int = 2, wgm: int = 8, split_m: int = 2, split_n: int = 2,
                grid_size: int = 256, xcd_chunk: int = 32):
    """mx4w made persistent: `grid_size` workgroups (one a CU), workgroup w
    takes blocks w, w + grid_size, ... and runs their K-tiles as one stream,
    so the ring's prologue for the next block is issued under the current
    block's last K-tiles and the store of a finished block sits between two
    K-tiles instead of ending the workgroup. One K-tile a loop iteration
    (the ring rotates at run time, as mx4w's rolled form). Same arguments and
    results as mx4w_kernel."""
    BKs = 128
    qm, qn = block_m // 2 // split_m, block_n // 2 // split_n
    if split_m not in (1, 2) or split_n not in (1, 2) or qm % 64 or qn % 64:
        raise ValueError(f"{block_m}x{block_n} split {split_m}x{split_n}: parts of 64-row multiples")
    order = [p for p in ((0, 0), (0, 1), (1, 1), (1, 0)) if p[0] < split_m and p[1] < split_n]
    ha, hb = split_m, split_n
    st_a = hk.ST(fp8e4m3, block_m // ha, BKs)
    st_b = hk.ST(fp8e4m3, block_n // hb, BKs)
    t_a = hk.RT(fp8e4m3, qm, BKs, "row")
    t_b = hk.RT(fp8e4m3, qn, BKs, "row")
    t_c = hk.RT(fp32, qm, qn, "col")

    def body(a: hk.Tensor[fp8e4m3], b: hk.Tensor[fp8e4m3], sa: hk.Tensor[i32],
             sb: hk.Tensor[i32], c: hk.Tensor[bf16], *, WGM: hk.Constexpr = wgm):
        g = hk.group(WARPS)
        As = [hk.lds.alloc(st_a, count=stages, name=f"As{h}") for h in range(ha)]
        Bs = [hk.lds.alloc(st_b, count=stages, name=f"Bs{h}") for h in range(hb)]
        A = [hk.zeros(t_a) for _ in range(split_m)]
        B = [hk.zeros(t_b) for _ in range(split_n)]
        C = [[hk.zeros(t_c) for _ in range(split_n)] for _ in range(split_m)]

        M = a.rows
        m_blocks = hk.cdiv(M, block_m)
        n_blocks = b.rows // block_n
        blocks = m_blocks * n_blocks
        wg = hk.arch.block_idx.x
        wid = hk.arch.warp_id()
        wr, wc = wid // 2, wid % 2
        k_tiles = a.cols // BKs
        G = hk.min(blocks, grid_size)
        mine = (blocks - wg + G - 1) // G                     # blocks this workgroup runs
        n_steps = mine * k_tiles

        def origin(i):
            """(m0, n0) of this workgroup's i-th block: the virtual id wg + i*G
            renumbered so a chunk of consecutive ids shares an XCD (as mx4w),
            then WGM block-rows a group."""
            t = hk.placement.renumber(wg + i * G, blocks, chunk=xcd_chunk, tail="stay")
            per_group = WGM * n_blocks
            first_m = (t // per_group) * WGM
            group_m = hk.min(m_blocks - first_m, WGM)
            return ((first_m + (t % per_group) % group_m) * block_m,
                    ((t % per_group) // group_m) * block_n)

        def a_row(i):
            return i * (block_m // split_m) + wr * qm

        def b_row(j):
            return j * (block_n // split_n) + wc * qn

        PIECES = [p for p in (("a", 0), ("b", 0), ("a", 1), ("b", 1))
                  if p[1] < (ha if p[0] == "a" else hb)]

        def piece(m0, n0, kt, u, kind, h):
            if kind == "a":
                g.load(As[h][u], a.at[m0 + h * (block_m // ha), kt * BKs], wait=False)
            else:
                g.load(Bs[h][u], b.at[n0 + h * (block_n // hb), kt * BKs], wait=False)

        S = [None, None]

        def load_scales(q, m0, n0, kt):
            old = S[q]
            S[q] = ([hk.load_mx_scales(sa, qm, 4, m0 + a_row(i), kt * 4,
                                       into=old[0][i] if old else None) for i in range(split_m)],
                    [hk.load_mx_scales(sb, qn, 4, n0 + b_row(j), kt * 4,
                                       into=old[1][j] if old else None) for j in range(split_n)])

        def hand_over():
            for i in range(split_m):
                S[1][0][i].copy(into=S[0][0][i])
            for jj in range(split_n):
                S[1][1][jj].copy(into=S[0][1][jj])

        def step(u, un, copy_at, scales_at):
            """One K-tile from ring role u; copy_at = (m0, n0, k) of the K-tile
            stages-1 ahead (into role un), scales_at = that of the next one."""
            g.wait_for(*[As[h][u] for h in range(ha)], *[Bs[h][u] for h in range(hb)])
            hk.arch.barrier()
            if scales_at is not None:
                load_scales(1, *scales_at)
            have = set()
            for p, (i, jj) in enumerate(order):
                new = [x for x in (("a", i), ("b", jj)) if x not in have]
                for kind, idx in new:
                    if kind == "a":
                        As[idx][u].subtile(qm, BKs, wr, 0).load(t_a, wait=False, into=A[idx])
                    else:
                        Bs[idx][u].subtile(qn, BKs, wc, 0).load(t_b, wait=False, into=B[idx])
                    have.add((kind, idx))
                if copy_at is not None:
                    for k in range(len(PIECES)):
                        if k % len(order) == p:
                            piece(*copy_at, un, *PIECES[k])
                if new:
                    hk.lds.wait_for(*[A[idx] if kind == "a" else B[idx] for kind, idx in new])
                hk.mma("mma_ABt", A[i].scaled(S[0][0][i]), B[jj].scaled(S[0][1][jj]), C[i][jj],
                       into=C[i][jj])
            hk.arch.sched_barrier()

        def finish_block(m0, n0):
            for i in range(split_m):
                for j in range(split_n):
                    c[hk.elem_coord(0, 0, m0 + a_row(i), n0 + b_row(j))].store(C[i][j], bounded=True)
                    C[i][j].zero_()

        def prologue(m0, n0):
            load_scales(0, m0, n0, 0)
            load_scales(1, m0, n0, 1)
            for t in range(stages - 1):
                for kind, h in PIECES:
                    piece(m0, n0, t, t, kind, h)

        # grid_size <= blocks (gemm_4p specializes it), so every workgroup has a
        # block. Blocks are an outer loop and K-tiles an inner one (a store
        # inside the K-tile loop made the compiler permute the accumulators every
        # K-tile); the next block's scales and first copies go out before this
        # block's store, so the store and the refill overlap.
        # The block is stored before the next one's scales and first copies go
        # out: issued under the last MFMAs (refill first) the scale loads landed
        # in VGPRs those still-queued scaled MFMAs had yet to read -- a 32-column
        # strip a block came out wrong (NaN), and nothing in the compiler waits
        # for it. Storing first reads every accumulator, which waits for the
        # MFMAs. (Moving the loads to the other scale set did not help: the
        # register allocator gives both sets the same VGPRs across the copy.)
        m0, n0 = origin(0)
        prologue(m0, n0)
        for bi in range(0, mine):
            rot = 0
            for kk in range(0, k_tiles - (stages - 1)):
                step(rot % stages, (rot + stages - 1) % stages, (m0, n0, kk + stages - 1),
                     (m0, n0, kk + 1))
                hand_over()
                rot = (rot + 1) % stages
            for d in range(stages - 1):
                last = d == stages - 2
                step((rot + d) % stages, None, None,
                     None if last else (m0, n0, k_tiles - (stages - 1) + d + 1))
                if not last:
                    hand_over()
            # The last block refills with itself (clamped): wasted copies, no branch.
            nm0, nn0 = origin(hk.min(bi + 1, mine - 1))
            hk.arch.barrier()
            finish_block(m0, n0)
            prologue(nm0, nn0)
            m0, n0 = nm0, nn0
        hk.arch.vm_wait(0)

    body.__name__ = name
    return hk.kernel(body, arch="gfx950", warps=WARPS, name=name, max_vgprs=512,
                     grid=lambda p: (grid_size, 1, 1))


_MX4P = {}


def gemm_4p(a, b, sa_packed, sb_packed, out=None, **sched):
    """bf16 (M, N) = (sa . a)(sb . b)^T on mx4p_kernel (persistent)."""
    import torch  # noqa: PLC0415

    bm, bn = sched.get("block_m", 256), sched.get("block_n", 256)
    blocks = -(-a.shape[0] // bm) * (b.shape[0] // bn)
    sched = dict(sched, grid_size=min(sched.get("grid_size", 256), blocks))
    key = tuple(sorted(sched.items()))
    kern = _MX4P.get(key)
    if kern is None:
        name = "mxfp8_4p" + "".join(f"_{k_}{v}" for k_, v in key)
        kern = _MX4P[key] = mx4p_kernel(name, **sched)
    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)
    kern(a, b, sa_packed, sb_packed, out)
    return out


_MX4W = {}


def gemm_4w(a, b, sa_packed, sb_packed, out=None, **sched):
    """bf16 (M, N) = (sa . a)(sb . b)^T on mx4w_kernel; sa_packed / sb_packed
    from pack_scales."""
    import torch  # noqa: PLC0415

    key = tuple(sorted(sched.items()))
    kern = _MX4W.get(key)
    if kern is None:
        name = "mxfp8_4w" + "".join(f"_{k_}{v}" for k_, v in key)
        kern = _MX4W[key] = mx4w_kernel(name, **sched)
    m = a.shape[0]
    if out is None:
        out = torch.empty((m, b.shape[0]), device=a.device, dtype=torch.bfloat16)
    kern(a, b, sa_packed, sb_packed, out)
    return out


def tail_split(m: int, n: int, cus: int = 256, block_m: int = 256, block_n: int = 256,
               tail_block: int = 128):
    """The column split that ends the main kernel's grid on a whole wave: N1, the largest multiple
    of block_n with ceil(m / block_m) * N1 / block_n a multiple of `cus`, the rest (N - N1 columns)
    left to a second launch of tail_block-square blocks. None when there is no such N1 or the
    rest is not a whole number of tail blocks."""
    mb = -(-m // block_m)
    if (mb * (n // block_n)) % cus == 0:
        return None                                  # already whole waves
    for n1 in range(n - block_n, 0, -block_n):
        if (mb * (n1 // block_n)) % cus == 0 and (n - n1) % tail_block == 0:
            tail_tiles = -(-m // tail_block) * ((n - n1) // tail_block)
            # the rest must be one wave of tail blocks: more, and the small blocks' lower
            # efficiency costs more than the partial wave it removes (8192 x 16384: 0.96x)
            return n1 if tail_tiles <= cus else None
    return None


def gemm_4w_tail(a, b, sa_packed, sb_packed, out=None, *, n1=None, tail=None, **sched):
    """gemm_4w with the grid's last, partial wave recut: columns [0, n1) on `sched` (whole waves),
    columns [n1, N) on `tail` (smaller blocks: one wave across all CUs). Every output element is
    still one workgroup's full-K sum in the same order -- bit-identical to gemm_4w."""
    import torch  # noqa: PLC0415

    m, n = a.shape[0], b.shape[0]
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    if n1 is None:
        n1 = tail_split(m, n, block_m=sched.get("block_m", 256), block_n=sched.get("block_n", 256))
    if not n1 or n1 >= n:
        return gemm_4w(a, b, sa_packed, sb_packed, out, **sched)
    tail = dict(tail or {"block_m": 128, "block_n": 128, "split_m": 1, "split_n": 1, "stages": 3,
                         "rolled": True})
    gemm_4w(a, b[:n1], sa_packed, sb_packed[: n1 // 64], out[:, :n1], **sched)
    gemm_4w(a, b[n1:], sa_packed, sb_packed[n1 // 64:], out[:, n1:], **tail)
    return out



def gemm_8w_tail(a, b, sa_packed, sb_aiter, sb_packed, out=None, *, n1=None, tail=None, **sched):
    """gemm_8w with the grid's last, partial wave recut as gemm_4w_tail does: columns [0, n1) on
    gemm_8w (whole waves), the rest on gemm_4w's 128 x 128 blocks (packed B scales: sb_packed).
    Bit-identical to gemm_8w."""
    import torch  # noqa: PLC0415

    m, n = a.shape[0], b.shape[0]
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    if n1 is None:
        n1 = tail_split(m, n)
    if not n1 or n1 >= n:
        return gemm_8w(a, b, sa_packed, sb_aiter, out, **sched)
    tail = dict(tail or {"block_m": 128, "block_n": 128, "split_m": 1, "split_n": 1, "stages": 3,
                         "rolled": True})
    gemm_8w(a, b[:n1], sa_packed, sb_aiter[:n1], out[:, :n1], **sched)
    gemm_4w(a, b[n1:], sa_packed, sb_packed[n1 // 64:], out[:, n1:], **tail)
    return out


# -- activation quant: bf16 x -> FP8 e4m3 x + E8M0 scales in the layout the GEMM reads --------

QUANT_ROWS, QUANT_COLS = 64, 128


def quant_kernel(packed: bool = True):
    """bf16 x [M, K] -> (FP8 e4m3 q [M, K], scales): 64 x 128 blocks through LDS (the last one
    stops at row M) into
    hk.mx_quant_store. `packed`: int32 scales in hk.mx.pack_scales' layout (rows past M at the unit
    scale), else uint8 in hk.mx.aiter_scales' 32-row layout. Bit for bit hk.mx.quant (AITER's
    per_1x32 rule, which vLLM's mxfp8_e4m3_quantize also follows) and then that layout: one launch
    where vLLM's route took a quant and a pack."""
    st = hk.ST(bf16, QUANT_ROWS, QUANT_COLS)
    S = i32 if packed else i8

    def mx_quant(x: hk.Tensor[bf16], q: hk.Tensor[fp8e4m3], s: hk.Tensor[S]):
        S  # noqa: B018 -- named in the body: an annotation resolves only through a free variable
        g = hk.group(4)
        Xs = hk.lds.alloc(st, name="Xs")
        r0 = hk.arch.block_idx.y * QUANT_ROWS
        c0 = hk.arch.block_idx.x * QUANT_COLS
        # The last row block stops at x's last row; the rest of the tile reads zero. A whole-tile
        # copy read up to 63 rows past x, which faults where x ends its allocator segment (B1's
        # combined arm). mx_quant_store neither stores those rows nor reads them for the scales.
        g.load(Xs, x.at[r0, c0], wait=False, rows=hk.min(x.rows - r0, QUANT_ROWS))
        g.wait_for(Xs)
        hk.arch.barrier()
        hk.mx_quant_store(Xs, q, s, r0, c0)

    name = "mxfp8_quant_" + ("packed" if packed else "aiter")
    mx_quant.__name__ = name
    return hk.kernel(mx_quant, arch="gfx950", warps=4, name=name,
                     grid=lambda p: (p.x.cols // QUANT_COLS, hk.cdiv(p.x.rows, QUANT_ROWS), 1))


_QUANT = {}


def quant(x, packed: bool = True):
    """(q, scales) for bf16 x [M, K], K a multiple of 128: scales packed ([ceil(M/64), K/128 * 64]
    int32, hk.mx.pack_scales of hk.mx.quant's) or in AITER's 32-row layout (uint8, M a multiple
    of 32)."""
    import torch  # noqa: PLC0415

    m, k = x.shape
    if k % QUANT_COLS:
        raise ValueError(f"mxfp8.quant: K={k}, a multiple of {QUANT_COLS}")
    q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    if packed:
        s = torch.empty((-(-m // 64), k // 128 * 64), device=x.device, dtype=torch.int32)
    else:
        if m % 32:
            raise ValueError(f"mxfp8.quant(packed=False): M={m}, a multiple of 32")
        kb = k // 32 + (-(k // 32)) % 8
        # the kernel writes every scale of x's own blocks; AITER's K pad (to 8 blocks) is unit scales
        s = (torch.empty if kb == k // 32 else torch.full)(
            *(((m, kb),) if kb == k // 32 else ((m, kb), 127)), device=x.device, dtype=torch.uint8)
    kern = _QUANT.get(packed)
    if kern is None:
        kern = _QUANT[packed] = quant_kernel(packed)
    kern(x, q, s.view(torch.int32) if packed else s.view(torch.int8))
    return q, s
