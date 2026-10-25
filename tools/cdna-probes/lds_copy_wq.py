"""An MXFP8 GEMM that ROCm 10's compiler turns wrong without a spill: hk's wait gate refuses it.

`wq_kernel` is the 4-wave MXFP8 schedule written by experiment 2's hk arm (KDA, 2026-10-25), kept
verbatim from its workspace. It is a port of HipKittens' CDNA4 MXFP8 kernel to hk-dsl, with two
LDS halves, four quadrant accumulators, and A's E8M0 scales read raw (`a_loose`).

The failure needs all of: block 128x256, `a_loose`, and K=1152 (KT 9, odd). ROCm 10's hipcc
(clang 23) then sits at 256 VGPRs and copies a B fragment into AGPRs with
`v_accvgpr_write a32, v58`. That copy comes right after the inline-asm `ds_read_b128 v[58:61]`
and before hk's lgkmcnt wait. The two columns of 16 that B base tile feeds came out wrong, 0.83
max-rel against the same kernel with packed scales. Nothing spills (ScratchSize 0). K=1024 and
K=1280 are exact, and so is 128x128.

    HK_ARCH=gfx950 python3 tools/cdna-probes/lds_copy_wq.py     # build both K, report the gate
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import hk  # noqa: E402
from hk import bf16, fp8e4m3, fp32, i8, i32  # noqa: E402

KS = 128            # one scaled MFMA's K
NUM_XCDS = 8


def remap(wg, tiles_m, tiles_n, wgm):
    """Linear workgroup id -> (block row, block col). Workgroups are dealt to the 8 XCDs round
    robin; give each XCD a contiguous run of tile ids, then walk the tiles in groups of `wgm`
    block rows, so the workgroups an XCD runs at once share A rows and B columns in its L2."""
    total = tiles_m * tiles_n
    per, extra = total // NUM_XCDS, total % NUM_XCDS
    xcd = wg % NUM_XCDS
    tid = xcd * per + hk.min(xcd, extra) + wg // NUM_XCDS
    in_group = wgm * tiles_n
    first = (tid // in_group) * wgm
    gsz = hk.min(tiles_m - first, wgm)
    r = tid % in_group
    return first + r % gsz, r // gsz




def wq_kernel(BM: int, BN: int, KT: int, il: int = 1, a_loose: bool = False, wgm: int = 0,
              tc: bool = False, b_loose: bool = False):
    """c5: the c2 quadrant schedule with c3's interleave, A and B scales both in wpack's
    warp-ordered packed layout (A packed per call, B cached per weight): one dword load a 64 rows
    and K-tile for every tile shape, the half picked by sa_at / sb_at."""
    HM, HN = BM // 2, BN // 2
    QM, QN = BM // 4, BN // 4
    st_a, st_b = hk.ST(fp8e4m3, HM, KS), hk.ST(fp8e4m3, HN, KS)
    t_a, t_b = hk.RT(fp8e4m3, QM, KS), hk.RT(fp8e4m3, QN, KS)
    # tc: accumulate C^T = B A^T, whose lane holds four consecutive columns of a row of C, so
    # the epilogue stores 8 bytes at a time instead of 2
    t_c = hk.RT(fp32, QN, QM, "col") if tc else hk.RT(fp32, QM, QN, "col")
    n_dma = (HM + HN) * KS // (256 * 16)          # DMA instructions a thread per half copy
    n_sc = (2 * QM // 16 if a_loose else QM // 32) + (2 * QN // 16 if b_loose else QN // 32)

    def mx_wq(a: hk.Tensor[fp8e4m3], b: hk.Tensor[fp8e4m3], sa: hk.Tensor[i8 if a_loose else i32],
              sb: hk.Tensor[i8 if b_loose else i32], c: hk.Tensor[bf16]):
        g = hk.group(4)
        As = hk.lds.alloc(st_a, count=4, name="As")
        Bs = hk.lds.alloc(st_b, count=4, name="Bs")
        if wgm:
            m0, n0 = remap(hk.arch.block_idx.x, c.rows // BM, c.cols // BN, wgm)
            m0, n0 = m0 * BM, n0 * BN
        else:
            m0 = hk.arch.block_idx.y * BM
            n0 = hk.arch.block_idx.x * BN
        w = hk.arch.warp_id()
        wr, wc = w // 2, w % 2
        ra = [m0 + h * HM + wr * QM for h in range(2)]
        rb = [n0 + h * HN + wc * QN for h in range(2)]
        va, vb = m0 + wr * 2 * QM, n0 + wc * 2 * QN      # warp-ordered scale rows
        A = [hk.zeros(t_a) for _ in range(2)]
        B = [hk.zeros(t_b) for _ in range(2)]
        C = [[hk.zeros(t_c) for _ in range(2)] for _ in range(2)]
        if a_loose:
            # raw xs bytes, one register a 16-row tile and K-tile (no per-call repack)
            SA = [[hk.load_mx_scales(sa, QM, 4, ra[h], 0, loose=True) for h in range(2)]
                  for _ in range(2)]
        else:
            SA = [hk.load_mx_scales(sa, 2 * QM, 4, va, 0) for _ in range(2)]
        if b_loose:
            # B's scales a byte a row (the cached per-row expansion), for quadrants that are
            # not whole 32-row pairs of a 64-row packed group
            SB = [[hk.load_mx_scales(sb, QN, 4, rb[h], 0, loose=True) for h in range(2)]
                  for _ in range(2)]
        else:
            SB = [hk.load_mx_scales(sb, 2 * QN, 4, vb, 0) for _ in range(2)]

        def scales(t, u):
            if a_loose:
                for h in range(2):
                    hk.load_mx_scales(sa, QM, 4, ra[h], t * 4, loose=True, into=SA[u][h])
            else:
                hk.load_mx_scales(sa, 2 * QM, 4, va, t * 4, into=SA[u])
            if b_loose:
                for h in range(2):
                    hk.load_mx_scales(sb, QN, 4, rb[h], t * 4, loose=True, into=SB[u][h])
            else:
                hk.load_mx_scales(sb, 2 * QN, 4, vb, t * 4, into=SB[u])

        def copy(t, u, h):
            g.load(As[2 * u + h], a.at[m0 + h * HM, t * KS], wait=False)
            g.load(Bs[2 * u + h], b.at[n0 + h * HN, t * KS], wait=False)

        def read_a(u, h):
            As[2 * u + h].subtile(QM, KS, wr, 0).load(t_a, wait=False, into=A[h])

        def read_b(u, h):
            Bs[2 * u + h].subtile(QN, KS, wc, 0).load(t_b, wait=False, into=B[h])

        def mma(u, i, j):
            if b_loose:
                hk.mma_ABt_mx(A[i], B[j], C[i][j], SA[u], SB[u][j], into=C[i][j],
                              sa_at=(i * QM // 16, 0))
            elif a_loose:
                hk.mma_ABt_mx(A[i], B[j], C[i][j], SA[u][i], SB[u], into=C[i][j],
                              sb_at=(j * QN // 16, 0))
            elif tc:
                hk.mma_ABt_mx(B[j], A[i], C[i][j], SB[u], SA[u], into=C[i][j],
                              sa_at=(j * QN // 16, 0), sb_at=(i * QM // 16, 0))
            else:
                hk.mma_ABt_mx(A[i], B[j], C[i][j], SA[u], SB[u], into=C[i][j],
                              sa_at=(i * QM // 16, 0), sb_at=(j * QN // 16, 0))

        def spread(nv):
            if il and nv:
                for _ in range(nv):
                    hk.arch.sched_group_barrier("mma", il)
                    hk.arch.sched_group_barrier("vmem_read", 1)
                hk.arch.sched_barrier()

        def iteration(k, u, more1, more2):
            v = 1 - u
            hk.lds.wait_for(A[0], B[0])
            g.wait_for(As[2 * u + 1], Bs[2 * u + 1])
            hk.arch.barrier()
            read_b(u, 1)
            read_a(u, 1)
            if more2:
                copy(k + 2, u, 0)
            if more1:
                scales(k + 1, v)
            mma(u, 0, 0)
            hk.lds.wait_for(B[1])
            mma(u, 0, 1)
            spread((n_dma if more2 else 0) + (n_sc if more1 else 0))
            hk.lds.wait_for(A[1])
            if more1:
                g.wait_for(As[2 * v], Bs[2 * v])
            hk.arch.barrier()
            if more2:
                copy(k + 2, u, 1)
            if more1:
                read_a(v, 0)
            mma(u, 1, 0)
            if more1:
                read_b(v, 0)
            mma(u, 1, 1)
            spread(n_dma if more2 else 0)

        scales(0, 0)
        copy(0, 0, 0)
        copy(0, 0, 1)
        if KT > 1:
            copy(1, 1, 0)
            copy(1, 1, 1)
        g.wait_for(As[0], Bs[0])
        hk.arch.barrier()
        read_a(0, 0)
        read_b(0, 0)
        main = KT - 2
        for p in hk.range(main // 2):
            iteration(2 * p, 0, True, True)
            iteration(2 * p + 1, 1, True, True)
        for k in range(2 * (main // 2), KT):
            iteration(k, k % 2, k + 1 < KT, k + 2 < KT)
        for i in range(2):
            for j in range(2):
                c[hk.elem_coord(0, 0, ra[i], rb[j])] = hk.transpose(C[i][j]) if tc else C[i][j]

    mx_wq.__name__ = f"mx_wq_{BM}x{BN}_k{KT}_i{il}{'_al' if a_loose else ''}_g{wgm}{'_tc' if tc else ''}{'_bl' if b_loose else ''}"
    if wgm:
        grid = lambda p: ((p.c.cols // BN) * (p.c.rows // BM), 1, 1)  # noqa: E731
    else:
        grid = lambda p: (p.c.cols // BN, p.c.rows // BM, 1)  # noqa: E731
    return hk.kernel(mx_wq, arch="gfx950", warps=4, name=mx_wq.__name__, grid=grid)


def source(kt: int, bm: int = 128, bn: int = 256) -> str:
    from hk.codegen import scaffold  # noqa: PLC0415

    return scaffold.device(wq_kernel(bm, bn, kt, 1, True).trace())


def main() -> int:
    from hk.runtime import compile as hc  # noqa: PLC0415
    from hk.runtime.resources import ResourceError  # noqa: PLC0415

    for kt in (8, 9, 10):
        try:
            hc.build(source(kt), "gfx950", name=f"lds_copy_wq_k{kt}", kind="co")
            print(f"K={kt * KS}: built")
        except ResourceError as e:
            print(f"K={kt * KS}: refused -- {str(e).splitlines()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
