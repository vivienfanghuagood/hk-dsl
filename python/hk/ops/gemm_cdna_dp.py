"""bf16 GEMM on gfx950 for the shapes the pingpong GEMM has too few blocks for.

C = A·Bᵀ as hk.ops.gemm_cdna, with smaller blocks and four warps. At M = 1024,
N = 4096, a 256x256 block is 64 workgroups for an MI355X's 256 CUs; a 128x128
one is 256, and a 64x128 one 512. Four warps in a 2 x 2 grid each own one
(block_m / 2) x (block_n / 2) accumulator. Nothing is ping-ponged: the copies
run `stages - 1` K-tiles ahead in a ring of LDS buffers, and a K-tile is

    g.wait_for(buffer t)          # this warp's share of tile t has landed
    barrier                       # every warp's has; tile t-1 is read out
    copy tile t + stages - 1      # into tile t-1's buffer
    math on buffer t              # the warp's A and B slices, 32 deep

one barrier a K-tile, which is both the read-after-copy edge for tile t and
the copy-after-read edge for the buffer the copy overwrites; hk-lds-pipeline
checks both. The loop body is `stages` K-tiles, so every buffer index is a
Python int, and K is padded up to a whole number of bodies with tiles copied
with a row limit of 0: zeros, which add nothing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .. import dsl as hk
from ..types import bf16, fp32
from . import _epilogue
from .gemm_cdna import NUM_XCDS


def gemm_kernel(name: str, *, block_m: int = 128, block_n: int = 128,
                k_step: int = 64, stages: int = 3, warps_m: int = 2,
                warps_n: int = 2, wgm: int = 8, xcd_chunk: int = 32,
                setprio: bool = False, splits: int = 1, pipelined: bool = True,
                lds_shape: Optional[str] = None, nt: str = "",
                arch: Optional[str] = None,
                max_vgprs: int = 256, min_occupancy: int = 1, epilogue=None):
    """`epilogue` (hk.ops._epilogue) replaces C's store with
    hk.templates.gemm's hook; with splits it goes on the reduction instead.

    `splits` > 1 also cuts K across workgroups: workgroup (b, s) sums the
    s-th share of K and stores its fp32 partial to ws[s], (splits, M, N); a
    second kernel (reduce_kernel) adds the shares in order and stores C. Two
    launches, because ordering them inside one -- hk.arch.wg_arrive, as
    hk.ops.decode_gemm does -- measured 2-8x slower than not splitting on an
    MI355X. K must be a multiple of splits * k_step."""
    reg_m, reg_n = block_m // warps_m, block_n // warps_n
    warps = warps_m * warps_n
    threads = 64 * warps
    if reg_m % 16 or reg_n % 16:
        raise ValueError(f"{block_m}x{block_n} over {warps_m}x{warps_n} warps "
                         f"does not give whole 16-row accumulators")
    for what, rows in (("A", block_m), ("B", block_n)):
        if rows * k_step * 2 % (16 * threads):
            raise ValueError(f"a {rows}x{k_step} tile of {what} is not a whole number "
                             f"of 16-byte copies for {threads} threads")
    # A slice is what one LDS read takes of a K-tile: 32 deep, or a whole
    # 64-wide subtile of st_16x64, which addresses no narrower window.
    sk = 64 if lds_shape == "16x64" else 32
    if k_step % sk or stages < 2:
        raise ValueError(f"k_step={k_step}, stages={stages}: {sk}-deep slices, two buffers or more")
    slices = k_step // sk

    st_a = hk.ST(bf16, block_m, k_step)
    st_b = hk.ST(bf16, block_n, k_step)
    t_a = hk.RT(bf16, reg_m, sk, "row")
    t_b = hk.RT(bf16, reg_n, sk, "row")
    t_c = hk.RT(fp32, reg_m, reg_n, "col")

    @hk.func
    def gemm(a, b, c, WGM, k0, k_tiles, finish):
        g = hk.group(warps)
        As = hk.lds.alloc(st_a, count=stages, name="As", shape=lds_shape)
        Bs = hk.lds.alloc(st_b, count=stages, name="Bs", shape=lds_shape)
        A = [hk.zeros(t_a) for _ in range(2)]
        B = [hk.zeros(t_b) for _ in range(2)]
        C = [hk.zeros(t_c)]

        M = a.rows
        m_blocks = hk.cdiv(M, block_m)
        n_blocks = b.rows // block_n
        blocks = m_blocks * n_blocks
        wgid = hk.arch.block_idx.x
        # xcd_chunk 0: an eighth of the grid, so each XCD gets one contiguous
        # run of the WGM order and the blocks that share a B tile share an L2
        # -- below 8 x 32 blocks a fixed chunk of 32 renumbers nothing.
        chunk = hk.max(blocks // NUM_XCDS, 1) if xcd_chunk == 0 else xcd_chunk
        per_round = NUM_XCDS * chunk
        limit = (blocks // per_round) * per_round
        local = wgid // NUM_XCDS
        moved = ((local // chunk) * per_round + (wgid % NUM_XCDS) * chunk
                 + local % chunk)
        stays = hk.min(hk.max(wgid - limit, 0), 1)
        wgid = moved + stays * (wgid - moved)
        per_group = WGM * n_blocks
        first_m = (wgid // per_group) * WGM
        group_m = hk.min(m_blocks - first_m, WGM)
        m0 = (first_m + (wgid % per_group) % group_m) * block_m
        n0 = ((wgid % per_group) // group_m) * block_n

        wid = hk.arch.warp_id()
        wr = wid // warps_n
        wc = wid % warps_n
        n_tiles = stages * hk.cdiv(k_tiles, stages)

        def rows_left(t, rows):
            return (1 - hk.min(hk.max(t - k_tiles + 1, 0), 1)) * rows

        def copy(t, u, pad=True):
            g.load(As[u], a.at[m0, k0 + t * k_step], wait=False,
                   rows=rows_left(t, block_m) if pad else None, nontemporal="a" in nt)
            g.load(Bs[u], b.at[n0, k0 + t * k_step], wait=False,
                   rows=rows_left(t, block_n) if pad else None, nontemporal="b" in nt)

        def math(u):
            def read(s, slot):
                As[u].subtile(reg_m, sk, wr, s).load(t_a, wait=False, into=A[slot])
                Bs[u].subtile(reg_n, sk, wc, s).load(t_b, wait=False, into=B[slot])

            read(0, 0)
            for s in range(slices):
                if s + 1 < slices:
                    read(s + 1, (s + 1) % 2)
                hk.lds.wait_for(A[s % 2], B[s % 2])
                if setprio:
                    hk.arch.setprio(1)
                C[0] += A[s % 2] @ B[s % 2].T
                if setprio:
                    hk.arch.setprio(0)

        if not pipelined:
            for t in range(stages - 1):
                copy(t, t, pad=t > 0)

            def k_tile(t, u, last):
                g.wait_for(As[u], Bs[u])
                hk.arch.barrier()
                if not last:
                    copy(t + stages - 1, (u + stages - 1) % stages)
                math(u)

            for t0 in range(0, n_tiles - stages, stages):
                for u in range(stages):
                    k_tile(t0 + u, u, False)
            t0 = n_tiles - stages
            for u in range(stages):
                # The last body: what is left to copy is the one tile past
                # it, which there is none of from its second tile on.
                k_tile(t0 + u, u, u > 0)
            finish(C[0], m0 + wr * reg_m, n0 + wc * reg_n, blocks)
            return

        # Pipelined: tile t's operands are read during tile t-1, into the
        # other of two register sets, so a K-tile starts on its first slice
        # at once, crosses its barrier under it, and issues the copy and the
        # next tile's reads under the rest.
        RA = [[hk.zeros(t_a) for _ in range(slices)] for _ in range(2)]
        RB = [[hk.zeros(t_b) for _ in range(slices)] for _ in range(2)]

        def read_tile(u, q):
            for sl in range(slices):
                As[u].subtile(reg_m, sk, wr, sl).load(t_a, wait=False, into=RA[q][sl])
                Bs[u].subtile(reg_n, sk, wc, sl).load(t_b, wait=False, into=RB[q][sl])

        def mma(q, sl):
            if setprio:
                hk.arch.setprio(1)
            C[0] += RA[q][sl] @ RB[q][sl].T
            if setprio:
                hk.arch.setprio(0)

        body_len = stages * 2 // (2 if stages % 2 == 0 else 1)
        n_tiles = body_len * hk.cdiv(k_tiles, body_len)

        for t in range(stages):
            copy(t, t, pad=t > 0)
        g.wait_for(As[0], Bs[0])
        hk.arch.barrier()
        read_tile(0, 0)

        def k_tile(t, j, copies, reads):
            u, q = j % stages, j % 2
            hk.lds.wait_for(RA[q][0], RB[q][0])
            mma(q, 0)
            # Every read of this tile retired before the barrier, which is
            # what lets the copy after it overwrite the buffer.
            hk.lds.wait_for(*RA[q][1:], *RB[q][1:]) if slices > 1 else None
            if reads:
                g.wait_for(As[(j + 1) % stages], Bs[(j + 1) % stages])
            hk.arch.barrier()
            if copies:
                copy(t + stages, u)
            if reads:
                read_tile((j + 1) % stages, 1 - q)
            for sl in range(1, slices):
                mma(q, sl)

        for t0 in range(0, n_tiles - body_len, body_len):
            for j in range(body_len):
                k_tile(t0 + j, j, True, True)
        t0 = n_tiles - body_len
        for j in range(body_len):
            k_tile(t0 + j, j, j + stages < body_len, j + 1 < body_len)

        finish(C[0], m0 + wr * reg_m, n0 + wc * reg_n, blocks)

    def body(a: hk.Tensor[bf16], b: hk.Tensor[bf16], c: hk.Tensor[bf16], *,
             WGM: hk.Constexpr = wgm, **ops):
        def finish(acc, r, col, blocks):
            if epilogue is None:
                c[hk.elem_coord(0, 0, r, col)].store(acc, bounded=True)
            else:
                epilogue.store(c, ops, acc, r, col)

        gemm(a, b, c, WGM, 0, a.cols // k_step, finish)

    def body_split(a: hk.Tensor[bf16], b: hk.Tensor[bf16], ws: hk.Tensor[fp32], *,
                   WGM: hk.Constexpr = wgm):
        s = hk.arch.block_idx.y
        per = a.cols // splits

        def finish(acc, r, col, blocks):
            ws[hk.elem_coord(0, s, r, col)].store(acc, bounded=True)

        gemm(a, b, ws, WGM, s * per, per // k_step, finish)

    if splits == 1:
        fn = _epilogue.sign(body, epilogue)
    elif epilogue is not None:
        raise ValueError("a split kernel stores fp32 shares; its epilogue goes "
                         "on the reduction (reduce)")
    else:
        fn = body_split
    fn.__name__ = name

    def _grid(p):
        return (hk.cdiv(p.a.rows, block_m) * (p.b.rows // block_n), splits, 1)

    return hk.kernel(fn, arch=arch, warps=warps, grid=_grid, name=name,
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)


DEFAULT: Dict[str, Any] = dict(block_m=128, block_n=128, k_step=64, stages=3,
                               warps_m=2, warps_n=2, wgm=8, xcd_chunk=32,
                               setprio=False, splits=1, pipelined=True,
                               lds_shape=None, nt="")


def _sched_name(s: Dict[str, Any]) -> str:
    return ("gemm_cdna_dp_{block_m}x{block_n}x{k_step}_s{stages}_w{warps_m}x{warps_n}"
            "_g{wgm}_x{xcd_chunk}").format(**s) + ("_sp" if s.get("setprio") else "") + (
                "" if s.get("pipelined", True) else "_np") + (
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


def reduce_kernel(name: str, splits: int, arch: Optional[str] = None):
    """C = bf16(ws[0] + ... + ws[splits-1]), ws (splits, M, N) fp32: four
    warps, a 16 x 128 strip each, shares added in order."""
    t = hk.RT(fp32, 16, 128, "row")

    def body(ws: hk.Tensor[fp32], c: hk.Tensor[bf16]):
        r = hk.arch.block_idx.y * 64 + hk.arch.warp_id() * 16
        col = hk.arch.block_idx.x * 128
        total = hk.zeros(t)
        for j in range(splits):
            total += ws[hk.elem_coord(0, j, r, col)].load(t, bounded=True)
        c[hk.elem_coord(0, 0, r, col)].store(total, bounded=True)

    body.__name__ = name
    return hk.kernel(body, arch=arch, warps=4, name=name,
                     grid=lambda p: (p.c.cols // 128, hk.cdiv(p.c.rows, 64), 1))


#: The strip reduce's plan for a contiguous output: four warps of two 16-byte
#: chunks each, 2048 fp32 a workgroup, so a 128 x 4096 output is 256
#: workgroups.
STRIP_REDUCE_WARPS, STRIP_REDUCE_CHUNKS = 4, 2


def strip_reduce_kernel(name: str, splits: int, warps: int = STRIP_REDUCE_WARPS,
                        arch: Optional[str] = None, epilogue=None):
    """reduce_kernel on strips: ws (splits, R, C) fp32 and c (R, C) bf16, a
    grid row a row of c, a lane's RUN fp32 of each share one load, the sum
    rounded to nearest even. A contiguous output is one row, M*N long.
    reduce_kernel's 16 x 128 tiles put 64 workgroups on a 128 x 4096 output:
    9.2 us to hipBLASLt's 4.4 for its own eight shares on an MI355X; this
    puts 256 there, 4.3 us.

    `epilogue` (hk.ops._epilogue): hk.templates.gemm's hook, on the sum --
    once, after every share is in, which is what makes a split-K GEMM right
    for an epilogue that is not linear. Its coordinates are C's, so the
    output is never the flat view."""

    def body(ws: hk.Tensor[fp32], c: hk.Tensor[bf16], *, CW=512, RUN=4, EXACT=1,
             **ops):
        r = hk.arch.block_idx.y
        c0 = (hk.arch.block_idx.x * warps + hk.arch.warp_id()) * CW
        s = hk.RS(fp32, 1, CW, RUN)
        total = ws.at[0, r, c0].load(s, bounded=not EXACT)
        for j in range(1, splits):
            total += ws.at[j, r, c0].load(s, bounded=not EXACT)
        if epilogue is None:
            c.at[r, c0].store(total, bounded=not EXACT)
        else:
            epilogue.store(c, ops, total, r, c0, bounded=not EXACT)

    body.__name__ = name
    _epilogue.sign(body, epilogue)
    return hk.kernel(body, arch=arch, warps=warps, name=name,
                     grid=lambda p: (hk.cdiv(p.c.cols, warps * p.CW), p.c.rows, 1))


_SCRATCH: Dict[Any, Any] = {}


def scratch_for(device, splits: int, m: int, n: int):
    """The fp32 (splits, M, N) a split launch writes its shares to, made once
    per shape: launches on one stream run in order."""
    import torch  # noqa: PLC0415

    key = (str(device), splits, m, n)
    if key not in _SCRATCH:
        _SCRATCH[key] = torch.empty((splits, m, n), device=device, dtype=torch.float32)
    return _SCRATCH[key]


def launch(sched: Dict[str, Any], a, b, out):
    """Run the schedule's kernel on (a, b) into out: one launch, or with
    splits the GEMM into its scratch and the reduce into out."""
    s = {**DEFAULT, **sched}
    k = kernel_for(s)
    if s["splits"] == 1:
        return k(a, b, out)
    m, n = out.shape
    ws = scratch_for(a.device, s["splits"], m, n)
    k(a, b, ws)
    return reduce(ws, out)


def _reduce_epilogue(ws, out, tgt, epilogue, operands, consts):
    """The reduction with hk.templates.gemm's epilogue on the sum: a row of
    strips a grid row (the epilogue's coordinates are C's), up to four
    chunks of runs a warp and four warps a workgroup."""
    if not tgt.strips:
        raise NotImplementedError(f"a split GEMM's epilogue runs on the strip "
                                  f"reduction, which {tgt.arch} has not got")
    splits = ws.shape[0]
    n = out.shape[1]
    run = 4 if n % (4 * tgt.wave) == 0 else 2
    cw = run * tgt.wave * max(1, min(4, n // (run * tgt.wave)))
    warps = max(1, min(4, n // cw))
    name = _epilogue.name(f"gemm_cdna_dp_sreduce_k{splits}_w{warps}", epilogue)
    if name not in _KERNELS:
        _KERNELS[name] = strip_reduce_kernel(name, splits, warps, epilogue=epilogue)
    _KERNELS[name](ws, out, *operands, CW=cw, RUN=run,
                   EXACT=int(n % (warps * cw) == 0), **consts)
    return out


def reduce(ws, out, strips=None, epilogue=None, operands=(), **consts):
    """out = bf16 of the sum of ws's shares, in order: on strips where the
    target has them (strip_reduce_kernel) -- over the flat view if out is
    contiguous, else a row of strips a row, one warp a row of up to a
    chunk of runs of two -- and on reduce_kernel's tiles otherwise.

    With `epilogue` (hk.templates.gemm's), out = epilogue(sum) in its dtype,
    the epilogue's tensors `operands` and constexprs `consts` passed on."""
    from ..target import current_arch, get_target  # noqa: PLC0415

    splits = ws.shape[0]
    m, n = out.shape
    tgt = get_target(current_arch())
    if epilogue is not None:
        return _reduce_epilogue(ws, out, tgt, epilogue, operands, consts)
    flat = out.is_contiguous()
    if strips is None:
        strips = tgt.strips and (flat or (out.stride(1) == 1 and n % (2 * tgt.wave) == 0))
    if strips:
        warps = STRIP_REDUCE_WARPS if flat else 1
        name = f"gemm_cdna_dp_sreduce_k{splits}" + ("" if flat else "_w1")
        if name not in _KERNELS:
            _KERNELS[name] = strip_reduce_kernel(name, splits, warps)
        if flat:
            cw = STRIP_REDUCE_CHUNKS * tgt.wave * 4
            _KERNELS[name](ws.view(splits, 1, m * n), out.view(1, m * n), CW=cw, RUN=4,
                           EXACT=int(m * n % (warps * cw) == 0))
        else:
            cw = 2 * tgt.wave * max(1, min(4, n // (2 * tgt.wave)))
            _KERNELS[name](ws, out, CW=cw, RUN=2, EXACT=int(n % cw == 0))
        return out
    name = f"gemm_cdna_dp_reduce_k{splits}"
    if name not in _KERNELS:
        _KERNELS[name] = reduce_kernel(name, splits)
    _KERNELS[name](ws, out)
    return out


__all__ = ["gemm_kernel", "reduce_kernel", "kernel_for", "launch", "scratch_for",
           "DEFAULT"]
