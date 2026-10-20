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
from ..types import bf16, fp32

#: An MI355X has eight XCDs; workgroups are dealt to them round robin.
NUM_XCDS = 8


def gemm_kernel(name: str, *, block_m: int = 256, block_n: int = 256,
                k_step: int = 64, warps_m: int = 2, warps_n: int = 4,
                wgm: int = 8, xcd_chunk: int = 64, setprio: bool = True,
                transposed_c: bool = False, early_store: bool = True,
                hipkittens_waits: bool = False, late_b: bool = True,
                splits: int = 1, lds_shape: Optional[str] = None, nt: str = "",
                arch: Optional[str] = None,
                max_vgprs: int = 256, min_occupancy: int = 1):
    """HipKittens' tiling by default: 256x256x64, eight warps as 2 x 4.

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

    @hk.func
    def gemm(a, b, WGM, k0, k_count, out):
        g = hk.group(warps)
        # Buffer u, half h of A is As[2u + h]: every index below is a Python
        # int, so each copy and read names its buffer.
        As = hk.lds.alloc(st_a, count=4, name="As", shape=lds_shape)
        Bs = hk.lds.alloc(st_b, count=4, name="Bs", shape=lds_shape)
        A = hk.zeros(t_a)
        B = [hk.zeros(t_b) for _ in range(2)]
        C = [[hk.zeros(t_c) for _ in range(2)] for _ in range(2)]

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
        per_group = WGM * n_blocks
        first_m = (wgid // per_group) * WGM
        group_m = hk.min(m_blocks - first_m, WGM)
        row = first_m + (wgid % per_group) % group_m
        col = (wgid % per_group) // group_m
        m0 = row * block_m
        n0 = col * block_n

        wid = hk.arch.warp_id()
        wr = wid // warps_n
        wc = wid % warps_n

        # The loop takes K-tiles in pairs. An odd count gets one more, past K,
        # copied with a row limit of 0: it reads zeros and its MFMAs add
        # nothing. The limit narrows the descriptor and moves no address --
        # moving the tile's row instead put the 64-bit address arithmetic of
        # every copy back in the loop, 60 VALU an iteration, 7% slower.
        n_tiles = 2 * hk.cdiv(k_count, 2)
        k_tiles = k_count

        # Only the copies of a pair's second tile can be the pad: they take
        # the limit, and the rest no arithmetic at all.
        def limit(t, rows, pad):
            return (1 - hk.min(hk.max(t - k_tiles + 1, 0), 1)) * rows if pad else None

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
            tile = hk.transpose(C[i][j]) if transposed_c else C[i][j]
            out(tile, m0 + i * half_m + wr * reg_m, n0 + j * half_n + wc * reg_n)

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
        with hk.arch.pingpong(wr == 1):
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

    def body(a: hk.Tensor[bf16], b: hk.Tensor[bf16], c: hk.Tensor[bf16], *,
             WGM: hk.Constexpr = wgm):
        def out(tile, r, col):
            c[hk.elem_coord(0, 0, r, col)].store(tile, bounded=True)

        gemm(a, b, WGM, 0, a.cols // k_step, out)

    # splits > 1: K cut across workgroups, each share to an fp32 scratch
    # (splits, M, N), and hk.ops.gemm_cdna_dp.reduce_kernel adds them.
    def body_split(a: hk.Tensor[bf16], b: hk.Tensor[bf16], ws: hk.Tensor[fp32], *,
                   WGM: hk.Constexpr = wgm):
        s = hk.arch.block_idx.y
        per = a.cols // splits

        def out(tile, r, col):
            ws[hk.elem_coord(0, s, r, col)].store(tile, bounded=True)

        gemm(a, b, WGM, s * per, per // k_step, out)

    fn = body if splits == 1 else body_split
    fn.__name__ = name

    def _grid(p):
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
                f"_nt{s['nt']}" if s.get("nt") else "")


_KERNELS: Dict[str, Any] = {}


def kernel_for(sched: Dict[str, Any]):
    s = {**DEFAULT, **sched}
    name = _sched_name(s)
    if name not in _KERNELS:
        _KERNELS[name] = gemm_kernel(name, **s)
    return _KERNELS[name]


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


def _routes() -> Dict[str, Dict[int, str]]:
    """The measured table, NxK -> {M: candidate}, read once."""
    if not _ROUTES:
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        p = Path(__file__).resolve().parents[1] / "tuned" / "routes_gemm_gfx950.json"
        if p.exists():
            doc = json.loads(p.read_text())
            for nk, by_m in doc.get("routes", {}).items():
                _ROUTES[nk] = {int(m): c for m, c in by_m.items()}
        _ROUTES.setdefault("", {})
    return _ROUTES


def fits(name: str, m: int, n: int, k: int) -> bool:
    """Whether candidate `name` takes the shape at all -- not whether it wins."""
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


def route(m: int, n: int, k: int) -> str:
    """The candidate for a shape: the measured table's, at the nearest M it
    measured for this N x K (on a log scale), if that candidate takes the
    shape; else the rule the table's measurements mostly follow."""
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
